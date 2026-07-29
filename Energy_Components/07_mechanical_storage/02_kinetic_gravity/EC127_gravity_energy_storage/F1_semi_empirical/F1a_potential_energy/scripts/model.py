"""
EC127 — Gravity Energy Storage — F1a Potential Energy Model

Stored gravitational potential energy:
    E_pot(h) = m * g * h                       [J]
    E_kwh    = m * g * h / 3.6e6               [kWh]

State of charge based on lift height:
    SOC = (h - h_min) / (h_max - h_min)

Charge (lifting mass with motor through drive train):
    P_elec_in = m_dot_kg_per_s * g * v_lift / (eta_motor * eta_drive)        [W]
    Or, given charging power, height update:
        dh/dt = P_elec * eta_motor * eta_drive / (m * g)

Discharge (lowering mass driving generator):
    P_elec_out = m * g * v_lower * eta_drive * eta_generator                  [W]
    Or, given discharging power, height update:
        dh/dt = -P_elec / (m * g * eta_drive * eta_generator)

Round-trip efficiency:
    eta_RT = (eta_motor * eta_drive) * (eta_drive * eta_generator)
           = eta_motor * eta_drive^2 * eta_generator
    Typical: 0.80-0.90.

Energy capacity:
    E_cap_kwh = m * g * (h_max - h_min) / 3.6e6

References:
    Botha, C.D., Kamper, M.J. (2019). Capability study of dry gravity energy storage.
    Journal of Energy Storage, 23, 159-174.
    Berrada, A., Loudiyi, K., Zorkani, I. (2017). System design and economic
    performance of gravity energy storage. Energy Conversion and Management, 137, 191-200.
"""

import numpy as np


class GravityF1a:
    """Gravity Energy Storage — potential energy model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.m = u["mass_kg"]["value"]                # kg
        self.h_max = u["h_max_m"]["value"]            # m
        self.h_min = u["h_min_m"]["value"]            # m
        self.g = u["g"]["value"]                      # m/s2
        self.P_rated = u["P_rated_kw"]["value"]       # kW
        self.eta_motor = u["eta_motor"]["value"]
        self.eta_drive = u["eta_drive"]["value"]
        self.eta_gen = u["eta_generator"]["value"]
        self._h_usable = self.h_max - self.h_min

    # ---------- state ----------
    def height(self, soc):
        """Mass height [m] at given SOC."""
        s = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        return self.h_min + s * self._h_usable

    def soc_from_height(self, h):
        h = np.asarray(h, dtype=float)
        return np.clip((h - self.h_min) / self._h_usable, 0.0, 1.0)

    def potential_energy_kwh(self, soc):
        """Stored potential energy [kWh] at given SOC."""
        h = self.height(soc)
        return self.m * self.g * h / 3.6e6

    # ---------- power ----------
    def charge_power(self, lift_velocity_mps):
        """Electrical input power [kW] to lift mass at velocity v [m/s]."""
        v = np.asarray(lift_velocity_mps, dtype=float)
        P_w = self.m * self.g * v / (self.eta_motor * self.eta_drive)
        return np.clip(P_w / 1000.0, 0.0, self.P_rated)

    def discharge_power(self, lower_velocity_mps):
        """Electrical output power [kW] when lowering mass at velocity v [m/s]."""
        v = np.asarray(lower_velocity_mps, dtype=float)
        P_w = self.m * self.g * v * self.eta_drive * self.eta_gen
        return np.clip(P_w / 1000.0, 0.0, self.P_rated)

    # ---------- capacity & efficiency ----------
    def energy_capacity_kwh(self):
        """Maximum usable energy capacity [kWh]."""
        return self.m * self.g * self._h_usable / 3.6e6

    def round_trip_efficiency(self):
        """eta = eta_motor * eta_drive^2 * eta_gen."""
        return self.eta_motor * (self.eta_drive ** 2) * self.eta_gen

    def charge_efficiency(self):
        return self.eta_motor * self.eta_drive

    def discharge_efficiency(self):
        return self.eta_drive * self.eta_gen

    # ---------- SOC update ----------
    def soc_update(self, soc0, power_kw, dt_hours, mode):
        """Update SOC under power command. Returns new SOC clamped to [0,1]."""
        s = float(np.clip(soc0, 0.0, 1.0))
        P = float(power_kw)
        dt = float(dt_hours)
        if mode == "idle" or P <= 0.0 or dt <= 0.0:
            return s
        # Power-limit clamp
        P = min(P, self.P_rated)
        if mode == "charge":
            E_elec_in = P * dt                         # kWh
            E_to_pot_kwh = E_elec_in * self.charge_efficiency()
            dE_J = E_to_pot_kwh * 3.6e6
            dh = dE_J / (self.m * self.g)
            h_new = self.height(s) + dh
            return float(self.soc_from_height(min(h_new, self.h_max)))
        elif mode == "discharge":
            E_elec_out = P * dt                        # kWh
            E_from_pot_kwh = E_elec_out / self.discharge_efficiency()
            dE_J = E_from_pot_kwh * 3.6e6
            dh = dE_J / (self.m * self.g)
            h_new = self.height(s) - dh
            return float(self.soc_from_height(max(h_new, self.h_min)))
        else:
            raise ValueError(f"mode must be 'charge', 'discharge', or 'idle', got '{mode}'")
