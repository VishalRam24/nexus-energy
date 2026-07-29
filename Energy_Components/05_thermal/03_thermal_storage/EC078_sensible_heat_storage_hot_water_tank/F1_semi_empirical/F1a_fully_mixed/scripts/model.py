"""
EC078 — Hot Water Tank TES — F1a Fully Mixed Model

Lumped-parameter (perfectly stirred) model:

    dT/dt = (Q_charge - Q_discharge - Q_loss) / (m * cp)

    Q_loss       = UA * (T - T_amb)
    energy_stored = m * cp * (T - T_ref)       [J]  (T_ref = T_min)
    soc          = (T - T_min) / (T_max - T_min)

Inputs at each call are the instantaneous operating point; time integration
is performed externally or in simulate.py via Euler steps.

Reference:
    Duffie, J.A., Beckman, W.A. (2013).
    Solar Engineering of Thermal Processes, 4th ed.
    John Wiley & Sons, ch. 8.
"""

import numpy as np


class HotWaterTankF1a:
    """Fully-mixed hot water tank — thermal state and energy balance."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V_L    = u["volume_L"]["value"]                # L
        self.m      = self.V_L * u["rho_water"]["value"]    # kg  (rho=1 kg/L)
        self.cp     = u["cp_water"]["value"]                # J/(kg·K)
        self.UA     = u["UA_loss"]["value"]                 # W/K
        self.T_min  = u["T_min"]["value"]                   # degC
        self.T_max  = u["T_max"]["value"]                   # degC
        self.T_ref  = self.T_min                            # reference for energy_stored

    # ------------------------------------------------------------------
    # Core model
    # ------------------------------------------------------------------

    def heat_loss(self, T, T_amb):
        """Standby heat loss to surroundings [W]. Positive when T > T_amb."""
        T     = np.asarray(T,     dtype=float)
        T_amb = np.asarray(T_amb, dtype=float)
        return self.UA * (T - T_amb)

    def dT_dt(self, T, q_charge, q_discharge, T_amb):
        """Rate of temperature change [K/s]."""
        T          = np.asarray(T,           dtype=float)
        q_charge   = np.asarray(q_charge,    dtype=float)
        q_discharge= np.asarray(q_discharge, dtype=float)
        T_amb      = np.asarray(T_amb,       dtype=float)
        Q_loss = self.heat_loss(T, T_amb)
        return (q_charge - q_discharge - Q_loss) / (self.m * self.cp)

    def energy_stored_kwh(self, T):
        """Thermal energy stored relative to T_ref [kWh]."""
        T = np.asarray(T, dtype=float)
        return self.m * self.cp * (T - self.T_ref) / 3.6e6   # J -> kWh

    def soc(self, T):
        """State of charge [0, 1] based on temperature."""
        T = np.asarray(T, dtype=float)
        return (T - self.T_min) / (self.T_max - self.T_min)
