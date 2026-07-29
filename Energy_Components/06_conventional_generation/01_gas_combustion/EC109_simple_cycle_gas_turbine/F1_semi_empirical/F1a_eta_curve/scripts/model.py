"""
EC109 — Simple Cycle Gas Turbine — F1a Efficiency Curve

eta(PLR, T_amb) = eta_rated * f_PLR(PLR) * f_amb(T_amb)
f_PLR = b0 + b1*PLR + b2*PLR^2       (peaked parabolic, max near PLR=1)
f_amb = 1 - f_amb_coeff*(T_amb - T_ref)   (derating with ambient heat)
P_out = P_rated * PLR  [MW]
P_fuel = P_out / eta   [MW_th]
fuel_rate = P_fuel / LHV_fuel * 1000  [kg/s]   (P_fuel in MW => *1000 kJ/s / kJ/kg)
heat_rate = 3600 / eta  [kJ/kWh]

Reference:
    Walsh & Fletcher (2004), "Gas Turbine Performance", 2nd ed., Blackwell Science.
    GE LM6000 product sheet (indicative values).
"""

import numpy as np


class SimpleCycleGasTurbineF1a:
    """Simple-cycle gas turbine — efficiency as a function of PLR and ambient temperature."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated     = u["P_rated"]["value"]        # MW
        self.eta_rated   = u["eta_rated"]["value"]
        self.T_ref       = u["T_amb_ref"]["value"]      # degC
        self.b0          = u["b0"]["value"]
        self.b1          = u["b1"]["value"]
        self.b2          = u["b2"]["value"]
        self.f_amb_coeff = u["f_amb_coeff"]["value"]    # 1/degC
        self.LHV         = u["LHV_fuel"]["value"]       # kJ/kg
        self.PLR_min     = u["PLR_min"]["value"]

    def f_plr(self, PLR):
        """Part-load correction factor (parabolic)."""
        PLR = np.asarray(PLR, dtype=float)
        return self.b0 + self.b1 * PLR + self.b2 * PLR ** 2

    def f_amb(self, T_amb):
        """Ambient temperature correction factor."""
        T = np.asarray(T_amb, dtype=float)
        return 1.0 - self.f_amb_coeff * (T - self.T_ref)

    def efficiency(self, PLR, T_amb):
        """Net LHV electrical efficiency."""
        PLR   = np.asarray(PLR,   dtype=float)
        T_amb = np.asarray(T_amb, dtype=float)
        eta = self.eta_rated * self.f_plr(PLR) * self.f_amb(T_amb)
        # Efficiency must be positive and physically bounded
        return np.clip(eta, 1e-6, 0.45)

    def power_mw(self, PLR):
        """Electrical output power in MW."""
        PLR = np.asarray(PLR, dtype=float)
        return self.P_rated * np.clip(PLR, 0.0, 1.0)

    def fuel_rate_kgs(self, PLR, T_amb):
        """Fuel mass flow rate in kg/s (natural gas)."""
        P_out = self.power_mw(PLR) * 1e3        # kW
        eta   = self.efficiency(PLR, T_amb)
        P_fuel_kw = P_out / eta                 # kW_th
        return P_fuel_kw / self.LHV             # kg/s

    def heat_rate_kjkwh(self, PLR, T_amb):
        """Heat rate in kJ/kWh (lower is better)."""
        eta = self.efficiency(PLR, T_amb)
        return 3600.0 / eta
