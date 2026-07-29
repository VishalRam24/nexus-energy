"""
EC107 -- Micro-CHP Stirling Engine -- F1b Part-Load + Ambient Temperature

Extends F1a by adding:
  1. Quadratic electrical efficiency curve vs PLR
     Stirling characteristic: moderate efficiency drop at part load
     (unlike SOFC, Stirling relies on temperature differential -> part-load
     operation at lower T_hot reduces efficiency more than SOFC)
  2. Thermal efficiency: very high and relatively flat
     (Stirling is optimised for heat; burner modulated to maintain set-point)
  3. Ambient temperature derating
     T_cold = ambient temperature affects Stirling cycle T_cold/T_hot ratio
     -> Carnot efficiency drops with rising ambient temp
  4. Heat-to-power ratio (HPR) derived from efficiency split

Key characteristics:
    - Very high heat-to-power ratio (HPR ~ 5-8 at rated)
    - Low electrical efficiency (10-15%) but very high thermal recovery (80-85%)
    - Total CHP efficiency: ~90-95% (most efficient heat delivery for gas-fired CHPs)
    - On/off control (not continuously modulated in residential units)
      -> F1b captures modulated PLR for commercial Stirling units

Stirling Carnot analogy:
    eta_Carnot = 1 - T_cold/T_hot
    At rated: T_hot ~ 650 degC, T_cold ~ 80 degC -> eta_Carnot ~ 63%
    Practical Stirling achieves ~15-25% of Carnot

Ambient derating:
    f_temp = 1 - 0.002 * max(0, T_amb - 25)
    (T_cold rises with ambient; ~0.2%/degC reduction in electrical output)

References:
    Hawkes, A. & Leach, M. (2007). Cost-effective operating strategy for residential
    micro-CHP. Energy, 32(5), 711-723.
    Lund, H. et al. (2016). Smart energy and flexibility. Elsevier.
    Kongtragool, B. & Wongwises, S. (2003). A review of solar-powered Stirling engines
    and low temperature differential Stirling engines. Renew. Sustain. Energy Rev., 7(2),
    131-154.
"""

import numpy as np


class MicroCHPStirlingF1b:
    """Stirling engine micro-CHP with part-load efficiency curve and ambient correction."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_el_rated   = u["P_el_rated_kw"]["value"]           # kW_e
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
        Stirling T_cold = ambient temperature; rising ambient reduces
        Carnot efficiency and hence electrical output.
        """
        T = np.asarray(T_amb_c, dtype=float)
        excess = np.maximum(0.0, T - self.temp_start)
        return np.clip(1.0 - self.temp_derate * excess, 0.5, 1.0)

    # ------------------------------------------------------------------
    # Efficiencies
    # ------------------------------------------------------------------

    def eta_electrical(self, PLR, T_amb_c=25.0):
        """Electrical efficiency [-].
        Stirling drops moderately at part load.
        """
        p     = np.asarray(PLR, dtype=float)
        f_plr = self.el_a + self.el_b * p + self.el_c * p ** 2
        f_T   = self.f_temperature(T_amb_c)
        eta   = self.eta_el_rated * f_plr * f_T
        return np.clip(eta, 0.0, 0.25)

    def eta_thermal(self, PLR):
        """Thermal efficiency [-].
        High and relatively flat; burner maintains heat output.
        """
        p     = np.asarray(PLR, dtype=float)
        f_plr = self.th_a + self.th_b * p
        eta   = self.eta_th_rated * f_plr
        return np.clip(eta, 0.0, 0.92)

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
