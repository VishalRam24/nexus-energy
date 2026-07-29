"""
EC083 — Borehole Thermal Energy Storage (BTES) — F1a Fully Mixed Model

Fully-mixed (0D) thermal model:
    E_stored = m * Cp * (T_store - T_initial)
    Q_loss = UA_loss * (T_store - T_amb)
    dE/dt = Q_charge - Q_discharge - Q_loss

Steady-state outputs at given T_store.

References:
    Nordell (1994). Borehole Heat Store Design Optimization.
    Hellström (1991). Ground Heat Storage: Thermal Analyses of Duct Storage Systems.
"""

import numpy as np


class BTESF1a:
    """Fully-mixed ground store model for BTES."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V = u["V_ground"]["value"]
        self.rho = u["rho"]["value"]
        self.Cp = u["Cp"]["value"]
        self.UA_loss = u["UA_loss"]["value"]
        self.T0 = u["T_initial"]["value"]
        self.T_max = u["T_max"]["value"]
        self.m = self.rho * self.V
        self.E_max = self.m * self.Cp * (self.T_max - self.T0)  # J

    def predict(self, T_store, T_amb=10.0, Q_charge=0.0, Q_discharge=0.0):
        T_store = np.asarray(T_store, dtype=float)
        T_amb = np.asarray(T_amb, dtype=float)
        Q_charge = np.asarray(Q_charge, dtype=float)
        Q_discharge = np.asarray(Q_discharge, dtype=float)

        E_stored_J = self.m * self.Cp * (T_store - self.T0)
        E_stored_J = np.maximum(E_stored_J, 0.0)

        Q_loss_W = self.UA_loss * (T_store - T_amb)
        Q_loss_W = np.maximum(Q_loss_W, 0.0)

        SOC = np.clip(E_stored_J / self.E_max, 0.0, 1.0)
        Q_net = Q_charge - Q_discharge - Q_loss_W

        return {
            "E_stored_MWh": E_stored_J / 3.6e9,
            "Q_loss_kW": Q_loss_W / 1000.0,
            "SOC": SOC,
            "Q_net_kW": Q_net / 1000.0,
            "E_max_MWh": self.E_max / 3.6e9,
        }
