"""
EC126 — Flywheel Energy Storage — F1b Thermal Model

Extends F1a kinetic model with detailed loss mechanisms:

1. Windage losses: P_windage = k_w * omega^3
   - Cubic with angular velocity (aerodynamic drag on rotor)
   - Scales with air density: k_w(T) = k_w_ref * rho_air(T)/rho_air_ref
   - Air density: rho(T) = rho_ref * T_ref_K / T_K (ideal gas)

2. Bearing losses: P_bearing = k_b * omega
   - Linear for magnetic bearings (eddy current losses)
   - Approximately temperature-independent for magnetic bearings

3. Self-discharge rate = (P_windage + P_bearing) / E_stored
   - Varies with SOC (higher speed = more windage loss but also more energy)

4. Temperature effects:
   - Higher ambient T -> lower air density -> less windage (good)
   - Vacuum containment reduces but doesn't eliminate windage

5. SOC proportional to omega^2 (kinetic energy):
   SOC = (omega^2 - omega_min^2) / (omega_max^2 - omega_min^2)

References:
    Arani et al. (2017). Energies, 10, 1361.
    Beacon Power (2011). Flywheel Energy Storage Technical Report.
    Genta, G. (2005). Kinetic Energy Storage. Butterworth-Heinemann.
"""

import numpy as np

_RPM_TO_RADS = 2.0 * np.pi / 60.0
_T_REF_K = 298.15  # 25C in Kelvin


class FlywheelF1b:
    """Flywheel energy storage — thermal model with speed-dependent losses."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.J = u["J_kgm2"]["value"]                     # kg*m2
        self.omega_max = u["omega_max_rpm"]["value"] * _RPM_TO_RADS  # rad/s
        self.omega_min = u["omega_min_rpm"]["value"] * _RPM_TO_RADS  # rad/s
        self.E_max = u["E_max_kwh"]["value"]               # kWh
        self.P_rated = u["P_rated_kw"]["value"]             # kW
        self.k_windage_ref = u["k_windage"]["value"]        # W/(rad/s)^3
        self.k_bearing = u["k_bearing"]["value"]            # W/(rad/s)
        self.T_ref = u["T_ref"]["value"]                    # degC
        self.rho_air_ref = u["rho_air_ref"]["value"]        # kg/m3
        self.eta_motor = u["eta_motor"]["value"]
        self.eta_gen = u["eta_gen"]["value"]

        self._omega2_range = self.omega_max ** 2 - self.omega_min ** 2

    def _soc_to_omega(self, soc):
        """Convert SOC to angular velocity [rad/s]."""
        soc = np.asarray(soc, dtype=float)
        omega2 = self.omega_min ** 2 + soc * self._omega2_range
        return np.sqrt(np.maximum(omega2, 0.0))

    def _omega_to_rpm(self, omega):
        return omega / _RPM_TO_RADS

    def air_density(self, ambient_temperature):
        """Air density [kg/m3] using ideal gas scaling."""
        T_amb = np.asarray(ambient_temperature, dtype=float)
        T_K = T_amb + 273.15
        return self.rho_air_ref * _T_REF_K / T_K

    def windage_coefficient(self, ambient_temperature):
        """Temperature-corrected windage coefficient."""
        rho = self.air_density(ambient_temperature)
        return self.k_windage_ref * rho / self.rho_air_ref

    def windage_loss(self, soc, ambient_temperature=25.0):
        """Windage loss [kW] = k_w(T) * omega^3 / 1000."""
        omega = self._soc_to_omega(soc)
        k_w = self.windage_coefficient(ambient_temperature)
        return k_w * omega ** 3 / 1000.0

    def bearing_loss(self, soc):
        """Bearing loss [kW] = k_b * omega / 1000."""
        omega = self._soc_to_omega(soc)
        return self.k_bearing * omega / 1000.0

    def total_standby_loss(self, soc, ambient_temperature=25.0):
        """Total standby losses [kW] = windage + bearing."""
        return self.windage_loss(soc, ambient_temperature) + self.bearing_loss(soc)

    def energy_stored(self, soc):
        """Energy stored [kWh] = 0.5 * J * omega^2 / 3.6e6."""
        omega = self._soc_to_omega(soc)
        return 0.5 * self.J * omega ** 2 / 3.6e6

    def self_discharge_rate(self, soc, ambient_temperature=25.0):
        """
        Self-discharge rate [1/h] = P_loss / E_stored.
        Higher at low SOC (less energy, but still losses from omega_min).
        """
        P_loss = self.total_standby_loss(soc, ambient_temperature)
        E = self.energy_stored(soc)
        return np.where(E > 1e-9, P_loss / E, 0.0)

    def speed_rpm(self, soc):
        """Rotor speed [rpm] at given SOC."""
        return self._omega_to_rpm(self._soc_to_omega(soc))

    def power_actual(self, soc, power_command_kw, ambient_temperature=25.0):
        """
        Actual power delivered/absorbed [kW], accounting for losses.

        power_command > 0: charging (electrical input)
        power_command < 0: discharging (electrical output, negative)

        Returns actual shaft power after motor/generator efficiency and standby losses.
        """
        P_cmd = np.asarray(power_command_kw, dtype=float)
        P_standby = self.total_standby_loss(soc, ambient_temperature)

        # Charging: electrical in -> shaft power
        P_shaft_charge = P_cmd * self.eta_motor - P_standby
        # Discharging: shaft power -> electrical out
        P_shaft_discharge = P_cmd * self.eta_gen - P_standby

        P_actual = np.where(P_cmd >= 0, P_shaft_charge, P_shaft_discharge)
        return P_actual

    def losses(self, soc, power_command_kw, ambient_temperature=25.0):
        """Total losses [kW] = |P_command| - |P_actual|."""
        P_cmd = np.asarray(power_command_kw, dtype=float)
        P_actual = self.power_actual(soc, P_cmd, ambient_temperature)
        return np.abs(P_cmd) - np.abs(P_actual)

    def efficiency(self, soc, power_command_kw, ambient_temperature=25.0):
        """
        One-way efficiency = |P_actual| / |P_command|.
        """
        P_cmd = np.asarray(power_command_kw, dtype=float)
        P_actual = self.power_actual(soc, P_cmd, ambient_temperature)
        P_cmd_abs = np.abs(P_cmd)
        P_cmd_safe = np.where(P_cmd_abs > 1e-6, P_cmd_abs, 1.0)
        return np.where(P_cmd_abs > 1e-6,
                        np.clip(np.abs(P_actual) / P_cmd_safe, 0.0, 1.0), 0.0)
