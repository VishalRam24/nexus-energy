"""
EC106 -- SOFC-Based Fuel Cell CHP -- F1b Part-Load + Ambient Temperature

Extends F1a by adding:
  1. Quadratic electrical efficiency curve vs PLR
     SOFC characteristic: relatively flat efficiency at part load (better than engines)
     because cell voltage rises slightly at lower current density.
  2. Thermal (waste heat) efficiency: inversely proportional to PLR
     (heat loss is a larger fraction of fuel at part load)
  3. Ambient temperature derating (small for SOFC vs gas engines;
     driven by air blower performance rather than combustion air density)
  4. Heat-to-power ratio (HPR) derived from efficiency split

Electrical efficiency:
    eta_el(PLR) = eta_el_rated * (a + b*PLR + c*PLR^2)
    Characteristic: stays high even at PLR=0.3 (SOFC advantage)

Thermal efficiency:
    eta_th(PLR) = eta_th_rated * (th_a + th_b*PLR)
    At part load thermal fraction rises (heat losses become proportionally larger)

Total efficiency:
    eta_total = eta_el + eta_th   [typical 0.75-0.90]

Ambient derating (above 25 degC):
    f_temp = 1 - 0.0015 * max(0, T_amb - 25)
    (smaller than gas engine 0.003/degC; SOFC electrochemistry not air-density limited)

Heat-to-power ratio:
    HPR = eta_th / eta_el

References:
    US DOE NETL, SOFC Program Review (2020). Fuel Cell R&D Program Overview.
    US EPA CHP Catalog (2017). Combined Heat and Power Technology Fact Sheets.
    Larminie, J. & Dicks, A. (2003). Fuel Cell Systems Explained, 2nd ed. Wiley.
    Braun, R.J. et al. (2006). Evaluation of fuel cell CHP at residential buildings.
    J. Power Sources, 158(2), 1290-1305.
"""

import numpy as np


class FuelCellCHPF1b:
    """SOFC-based fuel cell CHP with part-load efficiency curve and ambient correction."""

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
        """Temperature derating factor for rated power."""
        T = np.asarray(T_amb_c, dtype=float)
        excess = np.maximum(0.0, T - self.temp_start)
        return np.clip(1.0 - self.temp_derate * excess, 0.5, 1.0)

    # ------------------------------------------------------------------
    # Efficiencies
    # ------------------------------------------------------------------

    def eta_electrical(self, PLR, T_amb_c=25.0):
        """Electrical efficiency [-].
        SOFC maintains relatively flat profile at part load.
        """
        p = np.asarray(PLR, dtype=float)
        f_plr = self.el_a + self.el_b * p + self.el_c * p ** 2
        f_T   = self.f_temperature(T_amb_c)
        eta   = self.eta_el_rated * f_plr * f_T
        return np.clip(eta, 0.0, 0.65)

    def eta_thermal(self, PLR):
        """Thermal (waste heat recovery) efficiency [-].
        Higher relative to electrical at part load (proportional heat loss increases).
        """
        p     = np.asarray(PLR, dtype=float)
        f_plr = self.th_a + self.th_b * p
        eta   = self.eta_th_rated * f_plr
        return np.clip(eta, 0.0, 0.55)

    def eta_total(self, PLR, T_amb_c=25.0):
        """Total (first-law) efficiency = eta_el + eta_th."""
        return self.eta_electrical(PLR, T_amb_c) + self.eta_thermal(PLR)

    # ------------------------------------------------------------------
    # Power/heat flows
    # ------------------------------------------------------------------

    def power_electrical_kw(self, PLR, T_amb_c=25.0):
        """Electrical output [kW_e]."""
        p   = np.asarray(PLR, dtype=float)
        f_T = self.f_temperature(T_amb_c)
        return self.P_el_rated * p * f_T

    def fuel_input_kw(self, PLR, T_amb_c=25.0):
        """Fuel input power [kW_fuel]."""
        P_el   = self.power_electrical_kw(PLR, T_amb_c)
        eta_el = self.eta_electrical(PLR, T_amb_c)
        return np.where(eta_el > 1e-6, P_el / eta_el, 0.0)

    def heat_recovery_kw(self, PLR, T_amb_c=25.0):
        """Thermal heat recovery [kW_th]."""
        fuel   = self.fuel_input_kw(PLR, T_amb_c)
        eta_th = self.eta_thermal(PLR)
        return fuel * eta_th

    def heat_to_power_ratio(self, PLR, T_amb_c=25.0):
        """Heat-to-power ratio = Q_th / P_el."""
        eta_el = self.eta_electrical(PLR, T_amb_c)
        eta_th = self.eta_thermal(PLR)
        return np.where(eta_el > 1e-6, eta_th / eta_el, 0.0)
