"""
EC109 -- Simple Cycle Gas Turbine -- F1b Part-Load + Ambient Correction

Extends F1a by adding ISO-standard pressure/temperature corrections to both
power output and efficiency, plus a proper exhaust temperature model.

Part-load efficiency:
    f_PLR(PLR) = a + b*PLR + c*PLR^2
    Peak efficiency near PLR ~ -b/(2c) ~ 0.85 for default coefficients.

ISO ambient correction (power):
    P_corrected = P_iso * (P_amb / P_iso_ref) * sqrt(T_iso_ref / T_amb)

    - Higher ambient pressure  -> higher air density -> more mass flow -> more power
    - Higher ambient temperature -> lower air density -> less mass flow -> less power

Efficiency ambient correction:
    eta(PLR, T_amb) = eta_rated * f_PLR(PLR) * f_eta_T(T_amb)
    f_eta_T = (T_iso_ref / T_amb)^0.5  (compressor inlet density effect)

Exhaust temperature:
    T_exhaust ~ T_exhaust_rated + dT_partload * (1 - PLR)
    (At part load, TIT is reduced but expansion ratio drops -> exhaust temp
     can be slightly higher than at full load for some designs.)

Fuel flow:
    fuel_flow = P_out / (eta * LHV)  [kg/s]

Heat rate:
    HR = 3600 / eta  [kJ/kWh]

References:
    Walsh & Fletcher (2004), Gas Turbine Performance, 2nd ed., Blackwell Science.
    ISO 2314:2009 Gas turbines -- Acceptance tests.
    GE LM6000 product data.
"""

import numpy as np


class SimpleCycleGasTurbineF1b:
    """Simple-cycle gas turbine with part-load and ISO ambient corrections."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated_mw     = u["P_rated_mw"]["value"]
        self.eta_rated      = u["eta_rated"]["value"]
        self.T_iso_ref      = u["T_iso_ref_k"]["value"]         # K
        self.P_iso_ref      = u["P_iso_ref_kpa"]["value"]       # kPa
        self.a_plr          = u["a_plr"]["value"]
        self.b_plr          = u["b_plr"]["value"]
        self.c_plr          = u["c_plr"]["value"]
        self.LHV            = u["fuel_lhv_mj_kg"]["value"]      # MJ/kg
        self.PLR_min        = u["PLR_min"]["value"]
        self.T_exh_rated    = u["T_exhaust_rated_k"]["value"]   # K
        self.dT_partload    = u["exhaust_temp_rise_partload"]["value"]  # K

    # ------------------------------------------------------------------
    # Correction factors
    # ------------------------------------------------------------------

    def f_plr(self, PLR):
        """Part-load efficiency correction factor (quadratic).
        Peaks at PLR = -b/(2c) ~ 1.625 -> clamped to PLR<=1, so
        within [0.3, 1.0] the factor increases monotonically to ~1.0.
        """
        PLR = np.asarray(PLR, dtype=float)
        return self.a_plr + self.b_plr * PLR + self.c_plr * PLR ** 2

    def f_power_ambient(self, T_amb_k, P_amb_kpa):
        """ISO power correction factor.
        P_corrected / P_iso = (P_amb / P_ref) * sqrt(T_ref / T_amb)
        """
        T = np.asarray(T_amb_k, dtype=float)
        P = np.asarray(P_amb_kpa, dtype=float)
        return (P / self.P_iso_ref) * np.sqrt(self.T_iso_ref / T)

    def f_eta_ambient(self, T_amb_k):
        """Efficiency correction for ambient temperature.
        Higher inlet temperature degrades compressor performance.
        f_eta_T = (T_ref / T_amb)^0.5
        """
        T = np.asarray(T_amb_k, dtype=float)
        return np.sqrt(self.T_iso_ref / T)

    # ------------------------------------------------------------------
    # Primary outputs
    # ------------------------------------------------------------------

    def efficiency(self, PLR, T_amb_k, P_amb_kpa=101.325):
        """Net LHV electrical efficiency [-]."""
        PLR = np.asarray(PLR, dtype=float)
        eta = self.eta_rated * self.f_plr(PLR) * self.f_eta_ambient(T_amb_k)
        return np.clip(eta, 1e-6, 0.50)

    def power_output_kw(self, PLR, T_amb_k, P_amb_kpa=101.325):
        """Electrical power output [kW].
        P = P_rated_iso * PLR * f_power_ambient
        """
        PLR = np.asarray(PLR, dtype=float)
        f_amb = self.f_power_ambient(T_amb_k, P_amb_kpa)
        P_mw = self.P_rated_mw * PLR * f_amb
        return P_mw * 1e3  # kW

    def fuel_flow_kg_s(self, PLR, T_amb_k, P_amb_kpa=101.325):
        """Fuel mass flow rate [kg/s]."""
        P_kw = self.power_output_kw(PLR, T_amb_k, P_amb_kpa)
        eta = self.efficiency(PLR, T_amb_k, P_amb_kpa)
        # P_fuel [kW] = P_kw / eta;  fuel [kg/s] = P_fuel / (LHV * 1e3)
        P_fuel_kw = P_kw / eta
        return P_fuel_kw / (self.LHV * 1e3)

    def exhaust_temp_k(self, PLR):
        """Exhaust gas temperature [K].
        Slightly higher at part load due to reduced expansion ratio.
        """
        PLR = np.asarray(PLR, dtype=float)
        return self.T_exh_rated + self.dT_partload * (1.0 - PLR)

    def heat_rate_kj_kwh(self, PLR, T_amb_k, P_amb_kpa=101.325):
        """Heat rate [kJ/kWh] -- lower is better."""
        eta = self.efficiency(PLR, T_amb_k, P_amb_kpa)
        return 3600.0 / eta
