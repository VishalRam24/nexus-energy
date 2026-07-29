"""
EC108 -- Steam Turbine CHP -- F1b Part-Load + Ambient Temperature

Covers backpressure and extraction steam turbine configurations used in
industrial CHP. Steam is raised in a fired boiler (gas/biomass/waste heat)
and expanded through the turbine; exhaust or extracted steam provides heat.

Extends F1a by adding:
  1. Quadratic electrical efficiency curve vs PLR
     Steam turbines: slightly drops at part load (sliding-pressure operation
     maintains higher efficiency than throttle-governed units)
  2. Thermal extraction efficiency: relatively flat vs PLR
     (extraction steam pressure set-point holds thermal output proportional)
  3. Ambient temperature derating above 15 degC (ISO reference)
     Higher ambient temperature raises cooling-water temperature, which
     increases condenser back-pressure and reduces turbine expansion ratio
  4. Heat-to-power ratio (HPR) from efficiency split

Key characteristics:
    - High HPR (2-3 typical) — more heat than electricity
    - Lower electrical efficiency than gas turbine/engine CHP
    - Very high thermal efficiency (65-75%)
    - Total CHP efficiency: 85-95%
    - Often fired by industrial waste steam (reducing dependence on gas)

Thermal extraction:
    In backpressure mode, all steam exhausts at process pressure (P_back).
    In extraction mode, steam is partially extracted mid-turbine at process
    pressure, and remainder expands to condenser vacuum.
    F1b models net extraction thermal efficiency as function of PLR.

Ambient derating:
    f_temp = 1 - 0.003 * max(0, T_amb - 15)
    (condenser back-pressure effect; ISO reference 15 degC)

References:
    US EPA CHP Catalog (2017). Combined Heat and Power Technology Fact Sheets.
    Kehlhofer, R. et al. (2009). Combined-Cycle Gas & Steam Turbine Power Plants,
    3rd ed. PennWell.
    IEA (2008). Combined Heat and Power -- Evaluating the Benefits of Greater
    Global Investment.
"""

import numpy as np


class SteamTurbineCHPF1b:
    """Steam turbine CHP with part-load efficiency curve and ambient correction."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_el_rated   = u["P_el_rated_kw"]["value"]          # kW_e
        self.eta_el_rated = u["eta_el_rated"]["value"]
        self.eta_th_rated = u["eta_th_rated"]["value"]
        self.PLR_min      = u["PLR_min"]["value"]
        self.el_a         = u["el_plr_a"]["value"]
        self.el_b         = u["el_plr_b"]["value"]
        self.el_c         = u["el_plr_c"]["value"]
        self.th_a         = u["th_plr_a"]["value"]
        self.th_b         = u["th_plr_b"]["value"]
        self.T_ref        = u["T_ref_c"]["value"]
        self.temp_derate  = u["temp_derating_pct_per_degC"]["value"] / 100.0
        self.temp_start   = u["temp_derating_start_c"]["value"]

    # ------------------------------------------------------------------
    # Derating
    # ------------------------------------------------------------------

    def f_temperature(self, T_amb_c):
        """
        Temperature derating factor.
        Condenser back-pressure rises with ambient temperature.
        ISO reference is 15 degC (not 25 degC as for gas engines).
        """
        T = np.asarray(T_amb_c, dtype=float)
        excess = np.maximum(0.0, T - self.temp_start)
        return np.clip(1.0 - self.temp_derate * excess, 0.5, 1.0)

    # ------------------------------------------------------------------
    # Efficiencies
    # ------------------------------------------------------------------

    def eta_electrical(self, PLR, T_amb_c=15.0):
        """Electrical efficiency [-].
        Steam turbine: peaks near PLR~0.85-0.90 (sliding pressure optimum).
        """
        p     = np.asarray(PLR, dtype=float)
        f_plr = self.el_a + self.el_b * p + self.el_c * p ** 2
        f_T   = self.f_temperature(T_amb_c)
        eta   = self.eta_el_rated * f_plr * f_T
        return np.clip(eta, 0.0, 0.35)

    def eta_thermal(self, PLR):
        """Thermal (steam extraction) efficiency [-].
        Relatively flat vs PLR; extraction pressure set-point maintained.
        """
        p     = np.asarray(PLR, dtype=float)
        f_plr = self.th_a + self.th_b * p
        eta   = self.eta_th_rated * f_plr
        return np.clip(eta, 0.0, 0.80)

    def eta_total(self, PLR, T_amb_c=15.0):
        """Total (first-law) efficiency = eta_el + eta_th."""
        return self.eta_electrical(PLR, T_amb_c) + self.eta_thermal(PLR)

    # ------------------------------------------------------------------
    # Power/heat flows
    # ------------------------------------------------------------------

    def power_electrical_kw(self, PLR, T_amb_c=15.0):
        """Electrical output [kW_e]."""
        p   = np.asarray(PLR, dtype=float)
        f_T = self.f_temperature(T_amb_c)
        return self.P_el_rated * p * f_T

    def fuel_input_kw(self, PLR, T_amb_c=15.0):
        """Fuel (boiler thermal) input [kW_fuel]."""
        P_el   = self.power_electrical_kw(PLR, T_amb_c)
        eta_el = self.eta_electrical(PLR, T_amb_c)
        return np.where(eta_el > 1e-6, P_el / eta_el, 0.0)

    def heat_recovery_kw(self, PLR, T_amb_c=15.0):
        """Thermal heat output (steam extraction / exhaust) [kW_th]."""
        fuel   = self.fuel_input_kw(PLR, T_amb_c)
        eta_th = self.eta_thermal(PLR)
        return fuel * eta_th

    def heat_to_power_ratio(self, PLR, T_amb_c=15.0):
        """Heat-to-power ratio = Q_th / P_el."""
        eta_el = self.eta_electrical(PLR, T_amb_c)
        eta_th = self.eta_thermal(PLR)
        return np.where(eta_el > 1e-6, eta_th / eta_el, 0.0)
