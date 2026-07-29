"""
EC103 — Supercritical CO2 Brayton Cycle — F1a Efficiency Curve

Semi-empirical efficiency model for a recuperated supercritical CO2 (sCO2)
Brayton power cycle. The cycle is highly compact, achieves 45-50% thermal
efficiency at turbine inlet temperatures of 600-750°C, and operates at
high pressures (~20-30 MPa).

Model:
    eta_carnot = 1 - T_reject / T_in                  [Carnot bound, K]
    eta(PLR)   = eta_rated * (1 - a*(1-PLR)^2)
    eta(PLR<PLR_min) = 0
    eta        = min(eta, eta_carnot)                  [Carnot cap]
    P_out      = PLR * P_rated
    Q_in       = P_out / eta

Reference:
    Dostal, Driscoll & Hejzlar (2004), MIT-ANP-TR-100, "A Supercritical
        Carbon Dioxide Cycle for Next Generation Nuclear Reactors".
    Crespi et al. (2017), Appl. Energy 195, 152-183, "Supercritical
        carbon dioxide cycles for power generation: A review".
    Wright et al. (2010), Sandia SAND2010-0171, "Operation and Analysis
        of a Supercritical CO2 Brayton Cycle".
"""

import numpy as np


class SCO2BraytonF1a:
    """Recuperated sCO2 Brayton cycle — efficiency-curve model."""

    def __init__(self, params: dict):
        c = params["cycle"]
        self.P_rated    = c["P_rated"]["value"]      # W
        self.eta_rated  = c["eta_rated"]["value"]    # -
        self.a_partload = c["a_partload"]["value"]   # -
        self.T_in       = c["T_in"]["value"]         # degC
        self.T_reject   = c["T_reject"]["value"]     # degC
        self.P_high     = c["P_high"]["value"]       # MPa
        self.P_low      = c["P_low"]["value"]        # MPa
        self.PLR_min    = c["PLR_min"]["value"]      # -

    # ------------------------------------------------------------------

    def carnot_efficiency(self, T_hot_c, T_cold_c):
        T_hot  = np.asarray(T_hot_c,  dtype=float) + 273.15
        T_cold = np.asarray(T_cold_c, dtype=float) + 273.15
        return 1.0 - T_cold / T_hot

    def cycle_efficiency(self, PLR, T_in_c=None, T_reject_c=None):
        """sCO2 cycle thermal efficiency at the given operating point."""
        PLR = np.asarray(PLR, dtype=float)
        T_h = self.T_in     if T_in_c     is None else np.asarray(T_in_c,     dtype=float)
        T_c = self.T_reject if T_reject_c is None else np.asarray(T_reject_c, dtype=float)

        eta = self.eta_rated * (1.0 - self.a_partload * (1.0 - PLR) ** 2)
        eta_carnot = self.carnot_efficiency(T_h, T_c)
        eta = np.minimum(eta, eta_carnot)

        eta = np.where(PLR < self.PLR_min, 0.0, eta)
        eta = np.where(PLR <= 0.0, 0.0, eta)
        return np.clip(eta, 0.0, 1.0)

    def power_output(self, PLR):
        """Electrical power output [W]."""
        PLR = np.asarray(PLR, dtype=float)
        return np.where(PLR < self.PLR_min, 0.0,
                        np.clip(PLR, 0.0, 1.0)) * self.P_rated

    def heat_input(self, PLR, T_in_c=None, T_reject_c=None):
        """Thermal heat input from primary heater [W]."""
        P_out = self.power_output(PLR)
        eta = self.cycle_efficiency(PLR, T_in_c, T_reject_c)
        return np.where(eta > 0, P_out / eta, 0.0)

    def heat_rejected(self, PLR, T_in_c=None, T_reject_c=None):
        """Heat rejected at the precooler [W]."""
        return self.heat_input(PLR, T_in_c, T_reject_c) - self.power_output(PLR)
