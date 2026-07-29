"""
EC126 — Flywheel Energy Storage — F1a Kinetic Model

Equations:
    omega      = speed_rpm * 2*pi / 60                  [rad/s]
    E          = 0.5 * J * omega^2 / 3.6e6              [kWh]
    SOC        = (omega^2 - omega_min^2) / (omega_max^2 - omega_min^2)   [-]
    P_mech     = torque * omega                          [W] (+ = charge, - = discharge)
    P_elec     = P_mech * eta_motor   if charging  (P_mech > 0)
               = P_mech / eta_gen     if discharging (P_mech < 0)
    Self-discharge:  dE/dt = -k_sd * E  =>  E(t) = E0 * exp(-k_sd * t)
    RTE        = eta_motor * eta_gen * (1 - self_discharge_fraction)

Reference:
    Arani, A.A.K. et al. (2017). Review of Flywheel Energy Storage Systems
    Structures and Applications in Power Systems and Microgrids. Energies, 10, 1361.
"""

import numpy as np


class FlywheelF1a:
    """Flywheel energy storage — kinetic model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.J = u["J_kgm2"]["value"]                # kg·m²
        self.omega_max = u["omega_max_rads"]["value"] # rad/s
        self.omega_min = u["omega_min_rads"]["value"] # rad/s
        self.P_rated = u["P_rated_kw"]["value"]       # kW
        self.E_rated = u["E_rated_kwh"]["value"]      # kWh
        self.k_sd = u["k_sd_per_hr"]["value"]         # 1/h
        self.eta_motor = u["eta_motor"]["value"]      # -
        self.eta_gen = u["eta_gen"]["value"]           # -
        # Precompute for SOC
        self._denom = self.omega_max**2 - self.omega_min**2

    def _rpm_to_rads(self, speed_rpm):
        return np.asarray(speed_rpm, dtype=float) * 2 * np.pi / 60.0

    def energy_stored(self, speed_rpm):
        """Kinetic energy stored [kWh]."""
        omega = self._rpm_to_rads(speed_rpm)
        E_J = 0.5 * self.J * omega**2   # Joules
        return E_J / 3.6e6              # kWh

    def soc(self, speed_rpm):
        """
        State of charge [-], based on usable energy window.
        SOC = (omega^2 - omega_min^2) / (omega_max^2 - omega_min^2)
        """
        omega = self._rpm_to_rads(speed_rpm)
        raw = (omega**2 - self.omega_min**2) / self._denom
        return np.clip(raw, 0.0, 1.0)

    def mechanical_power(self, speed_rpm, torque_nm):
        """
        Mechanical power at the shaft [kW].
        P = T * omega. Positive = charging, negative = discharging.
        Clipped to ±P_rated.
        """
        omega = self._rpm_to_rads(speed_rpm)
        T = np.asarray(torque_nm, dtype=float)
        P_mech_W = T * omega
        return np.clip(P_mech_W / 1000.0, -self.P_rated, self.P_rated)

    def electrical_power(self, speed_rpm, torque_nm):
        """
        Electrical power at the terminals [kW].
        Charging: P_elec > 0 (more electrical input needed for mechanical output)
        Discharging: P_elec < 0 (less electrical output from mechanical)
        """
        P_mech = self.mechanical_power(speed_rpm, torque_nm)
        P_elec = np.where(
            P_mech >= 0,
            P_mech / self.eta_motor,   # charging: more electrical in
            P_mech * self.eta_gen,      # discharging: less electrical out
        )
        return P_elec

    def self_discharge_power(self, speed_rpm):
        """
        Self-discharge loss [kW] = k_sd * E_stored.
        Represents bearing friction, windage, and eddy current losses.
        """
        E = self.energy_stored(speed_rpm)
        return self.k_sd * E  # kWh/h = kW (energy per hour = average power)

    def round_trip_efficiency(self, time_hours=0.0):
        """
        Round-trip efficiency including self-discharge over a hold time.
        RTE = eta_motor * eta_gen * exp(-k_sd * time_hours)
        """
        t = np.asarray(time_hours, dtype=float)
        return self.eta_motor * self.eta_gen * np.exp(-self.k_sd * t)

    def energy_after_standby(self, initial_soc, time_hours):
        """
        Energy remaining [kWh] after standing idle for time_hours.
        E(t) = E0 * exp(-k_sd * t)
        """
        E0 = initial_soc * self.E_rated
        t = np.asarray(time_hours, dtype=float)
        return E0 * np.exp(-self.k_sd * t)
