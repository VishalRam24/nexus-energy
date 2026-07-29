"""
EC079 — Molten Salt Thermal Energy Storage — F1a Fully Mixed Model

Energy balance (0D fully mixed assumption):
    dT/dt = (Q_charge - Q_discharge - Q_loss) / (m * cp)

where:
    Q_loss = UA * (T - T_ambient)   [W]
    m      = 1,800,000 kg           (solar salt at 1800 kg/m3, 1000 m3)
    cp     = 1500 J/(kg·K)          (solar salt)

Solar salt composition: 60% NaNO3 + 40% KNO3
Operating range: 290°C (cold tank) to 565°C (hot tank)
Must stay above 220°C solidification point.

State-of-Charge (SOC):
    SOC = (T - T_cold) / (T_hot - T_cold)    in [0, 1]

Energy stored (relative to cold state):
    E_stored = m * cp * (T - T_cold)   [J]  → converted to MWh

Reference:
    Herrmann, U., Kelly, B., Price, H. (2004).
    "Two-tank molten salt storage for parabolic trough solar power plants."
    Energy, 29(5-6), 883-893.
"""

import numpy as np


class MoltenSaltTESF1a:
    """Fully mixed molten salt thermal energy storage (0D energy balance)."""

    def __init__(self, params: dict):
        t = params["tank"]
        self.mass      = t["mass_kg"]["value"]            # kg
        self.cp        = t["cp_J_kgK"]["value"]           # J/(kg·K)
        self.UA        = t["UA_loss_W_K"]["value"]        # W/K
        self.T_hot     = t["T_hot_C"]["value"]            # °C
        self.T_cold    = t["T_cold_C"]["value"]           # °C
        self.T_melt    = t["T_melt_C"]["value"]           # °C
        self.T_amb_ref = t["T_ambient_ref_C"]["value"]    # °C
        self.volume    = t["volume_m3"]["value"]          # m3

        # Total energy capacity [J]
        self.E_capacity_J = self.mass * self.cp * (self.T_hot - self.T_cold)
        # In MWh
        self.E_capacity_MWh = self.E_capacity_J / 3.6e9

    def dT_dt(self, T_c, q_charge_kw, q_discharge_kw, T_amb_c=None):
        """
        Rate of temperature change [K/s].

        Args:
            T_c:            Current salt temperature [°C]
            q_charge_kw:    Thermal power input (charging) [kW]
            q_discharge_kw: Thermal power output (discharging) [kW]
            T_amb_c:        Ambient temperature [°C], defaults to reference

        Returns:
            dT/dt [K/s]
        """
        if T_amb_c is None:
            T_amb_c = self.T_amb_ref

        Q_charge    = np.asarray(q_charge_kw,    dtype=float) * 1000.0  # W
        Q_discharge = np.asarray(q_discharge_kw, dtype=float) * 1000.0  # W
        T           = np.asarray(T_c,            dtype=float)
        T_amb       = np.asarray(T_amb_c,        dtype=float)

        Q_loss = self.UA * (T - T_amb)          # Heat loss [W]
        Q_net  = Q_charge - Q_discharge - Q_loss # Net heat [W]

        return Q_net / (self.mass * self.cp)    # K/s

    def energy_stored_mwh(self, T_c):
        """Energy stored relative to cold reference temperature [MWh]."""
        T = np.asarray(T_c, dtype=float)
        E_J = self.mass * self.cp * (T - self.T_cold)
        return np.maximum(E_J / 3.6e9, 0.0)

    def soc(self, T_c):
        """State of charge [0, 1] based on temperature relative to operating range."""
        T = np.asarray(T_c, dtype=float)
        return np.clip((T - self.T_cold) / (self.T_hot - self.T_cold), 0.0, 1.0)

    def heat_loss_kw(self, T_c, T_amb_c=None):
        """Instantaneous heat loss to environment [kW]."""
        if T_amb_c is None:
            T_amb_c = self.T_amb_ref
        T     = np.asarray(T_c,     dtype=float)
        T_amb = np.asarray(T_amb_c, dtype=float)
        return self.UA * (T - T_amb) / 1000.0  # kW

    def simulate(self, T0_c, q_charge_profile, q_discharge_profile,
                 T_amb_profile=None, dt_s=3600.0):
        """
        Time-step simulation using Euler integration.

        Args:
            T0_c:               Initial temperature [°C]
            q_charge_profile:   Array of charge power [kW] at each time step
            q_discharge_profile: Array of discharge power [kW]
            T_amb_profile:      Array of ambient temperature [°C] (or scalar)
            dt_s:               Time step [s], default 3600 (1 hour)

        Returns:
            dict of time arrays: T, soc, energy_mwh, heat_loss_kw, dT_dt
        """
        N = len(q_charge_profile)
        T_arr        = np.zeros(N + 1)
        T_arr[0]     = T0_c
        if T_amb_profile is None:
            T_amb_arr = np.full(N, self.T_amb_ref)
        else:
            T_amb_arr = np.asarray(T_amb_profile, dtype=float)

        dT_arr   = np.zeros(N)
        loss_arr = np.zeros(N)

        for i in range(N):
            q_c = q_charge_profile[i]
            q_d = q_discharge_profile[i]
            T_a = T_amb_arr[i]
            dt  = self.dT_dt(T_arr[i], q_c, q_d, T_a)
            dT_arr[i]   = dt
            loss_arr[i] = self.heat_loss_kw(T_arr[i], T_a)
            T_new       = T_arr[i] + dt * dt_s
            # Enforce physical limits (can't go below melt point or above hot tank design)
            T_arr[i + 1] = np.clip(T_new, self.T_melt, self.T_hot + 20.0)

        return {
            "T_C":          T_arr,
            "soc":          self.soc(T_arr),
            "energy_mwh":   self.energy_stored_mwh(T_arr),
            "heat_loss_kw": np.append(loss_arr, loss_arr[-1]),
            "dT_dt":        np.append(dT_arr, dT_arr[-1]),
        }
