"""
EC105 -- Gas Turbine CHP -- F1b Part-Load + Ambient + HRSG Exhaust Temperature

Extends F1a by adding:
  1. ISO ambient correction for power and efficiency (temperature + pressure)
  2. HRSG exhaust temperature model -- exhaust gas T rises at part load,
     which modifies HRSG heat recovery effectiveness
  3. Heat-to-power ratio as a function of PLR + exhaust conditions

Electrical efficiency:
    eta_el(PLR, T_amb, P_amb) = eta_el_rated * f_PLR(PLR) * f_amb(T_amb, P_amb)
    f_PLR(PLR) = a + b*PLR + c*PLR^2
    f_amb = (P_amb/P_ref) * sqrt(T_ref/T_amb)   [ISO air-density correction]

Power output:
    P_el = P_el_rated * PLR * f_amb_power(T_amb, P_amb)

Exhaust temperature:
    T_exh(PLR) = T_exh_rated + dT_offset * (1 - PLR)
    (At part load, TIT reduced but expansion ratio also reduced -> exhaust rises)

HRSG thermal efficiency (depends on exhaust temperature):
    eta_th(PLR, T_exh) = eta_th_rated * f_PLR_th(PLR) * f_T_exh(T_exh)
    f_PLR_th(PLR) = th_a + th_b * PLR
    f_T_exh(T_exh) = 1 + hrsg_eff_T_coeff * (T_exh - T_exh_rated)
    [Higher exhaust T -> HRSG recovers more heat up to pinch-point limit]

Heat recovery:
    Q_th = fuel_input * eta_th

Heat-to-power ratio:
    HPR = Q_th / P_el

References:
    US EPA CHP Catalog (2017). Combined Heat and Power Technology Fact Sheets.
    Kehlhofer et al. (2009). Combined-Cycle Gas & Steam Turbine Power Plants, 3rd ed.
    Walsh & Fletcher (2004). Gas Turbine Performance, 2nd ed.
    ISO 2314:2009 Gas turbines -- Acceptance tests.
"""

import numpy as np


class GasTurbineCHPF1b:
    """Gas turbine CHP with part-load, ambient correction, and HRSG exhaust T model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_el_rated       = u["P_el_rated_kw"]["value"]          # kW_e
        self.eta_el_rated     = u["eta_el_rated"]["value"]
        self.eta_th_rated     = u["eta_th_rated"]["value"]
        self.PLR_min          = u["PLR_min"]["value"]
        self.el_a             = u["el_plr_a"]["value"]
        self.el_b             = u["el_plr_b"]["value"]
        self.el_c             = u["el_plr_c"]["value"]
        self.th_a             = u["th_plr_a"]["value"]
        self.th_b             = u["th_plr_b"]["value"]
        self.T_ref            = u["T_amb_ref_k"]["value"]            # K
        self.P_ref            = u["P_amb_ref_kpa"]["value"]          # kPa
        self.derate_pct       = u["temp_derate_pct_per_k"]["value"] / 100.0
        self.T_exh_rated      = u["T_exhaust_rated_k"]["value"]      # K
        self.dT_exh_offset    = u["T_exhaust_partload_offset_k"]["value"]
        self.hrsg_T_coeff     = u["hrsg_eff_T_coeff"]["value"]

    # ------------------------------------------------------------------
    # Ambient correction factors (ISO)
    # ------------------------------------------------------------------

    def f_power_ambient(self, T_amb_k, P_amb_kpa=101.325):
        """
        ISO power correction factor for gas turbine (air-density effect).
        f = (P_amb / P_ref) * sqrt(T_ref / T_amb)
        """
        T = np.asarray(T_amb_k, dtype=float)
        P = np.asarray(P_amb_kpa, dtype=float)
        return (P / self.P_ref) * np.sqrt(self.T_ref / T)

    def f_eta_ambient(self, T_amb_k):
        """
        Efficiency correction for ambient temperature.
        Compressor inlet temperature affects pressure ratio and specific work.
        f_eta = sqrt(T_ref / T_amb)
        """
        T = np.asarray(T_amb_k, dtype=float)
        return np.sqrt(self.T_ref / T)

    # ------------------------------------------------------------------
    # PLR correction factors
    # ------------------------------------------------------------------

    def f_plr_electrical(self, PLR):
        """Quadratic PLR correction for electrical efficiency."""
        p = np.asarray(PLR, dtype=float)
        return self.el_a + self.el_b * p + self.el_c * p ** 2

    def f_plr_thermal(self, PLR):
        """Linear PLR correction for thermal efficiency (HRSG)."""
        p = np.asarray(PLR, dtype=float)
        return self.th_a + self.th_b * p

    # ------------------------------------------------------------------
    # Exhaust temperature
    # ------------------------------------------------------------------

    def exhaust_temp_k(self, PLR):
        """
        HRSG inlet exhaust gas temperature [K].
        At part load: TIT reduced but expansion ratio also drops,
        so exhaust temperature rises slightly above full-load value.
        T_exh(PLR) = T_exh_rated + dT_offset * (1 - PLR)
        """
        p = np.asarray(PLR, dtype=float)
        return self.T_exh_rated + self.dT_exh_offset * (1.0 - p)

    # ------------------------------------------------------------------
    # HRSG correction for exhaust temperature
    # ------------------------------------------------------------------

    def f_hrsg_temp(self, T_exh_k):
        """
        HRSG effectiveness modifier as a function of exhaust inlet temperature.
        Higher exhaust temperature -> higher log-mean temperature difference
        -> more heat transfer, up to pinch-point limits.
        f = 1 + coeff * (T_exh - T_exh_rated)
        Clipped to [0.8, 1.2] -- HRSG cannot capture unlimited heat.
        """
        T = np.asarray(T_exh_k, dtype=float)
        f = 1.0 + self.hrsg_T_coeff * (T - self.T_exh_rated)
        return np.clip(f, 0.8, 1.2)

    # ------------------------------------------------------------------
    # Efficiencies
    # ------------------------------------------------------------------

    def eta_electrical(self, PLR, T_amb_k=288.15, P_amb_kpa=101.325):
        """Electrical efficiency [-]."""
        p = np.asarray(PLR, dtype=float)
        f_plr = self.f_plr_electrical(p)
        f_eta = self.f_eta_ambient(T_amb_k)
        eta = self.eta_el_rated * f_plr * f_eta
        return np.clip(eta, 0.0, 0.50)

    def eta_thermal(self, PLR, T_amb_k=288.15):
        """Thermal (HRSG) efficiency [-], accounting for exhaust T at part load."""
        p = np.asarray(PLR, dtype=float)
        T_exh = self.exhaust_temp_k(p)
        f_plr = self.f_plr_thermal(p)
        f_T = self.f_hrsg_temp(T_exh)
        eta = self.eta_th_rated * f_plr * f_T
        return np.clip(eta, 0.0, 0.65)

    def eta_total(self, PLR, T_amb_k=288.15, P_amb_kpa=101.325):
        """Total (first-law) CHP efficiency = eta_el + eta_th."""
        return self.eta_electrical(PLR, T_amb_k, P_amb_kpa) + self.eta_thermal(PLR, T_amb_k)

    # ------------------------------------------------------------------
    # Power / heat flows
    # ------------------------------------------------------------------

    def power_electrical_kw(self, PLR, T_amb_k=288.15, P_amb_kpa=101.325):
        """Electrical output [kW_e]."""
        p = np.asarray(PLR, dtype=float)
        f_amb = self.f_power_ambient(T_amb_k, P_amb_kpa)
        return self.P_el_rated * p * f_amb

    def fuel_input_kw(self, PLR, T_amb_k=288.15, P_amb_kpa=101.325):
        """Fuel input power [kW_fuel, LHV]."""
        P_el = self.power_electrical_kw(PLR, T_amb_k, P_amb_kpa)
        eta_el = self.eta_electrical(PLR, T_amb_k, P_amb_kpa)
        return np.where(eta_el > 1e-6, P_el / eta_el, 0.0)

    def heat_recovery_kw(self, PLR, T_amb_k=288.15, P_amb_kpa=101.325):
        """HRSG thermal heat recovery [kW_th]."""
        fuel = self.fuel_input_kw(PLR, T_amb_k, P_amb_kpa)
        eta_th = self.eta_thermal(PLR, T_amb_k)
        return fuel * eta_th

    def heat_to_power_ratio(self, PLR, T_amb_k=288.15, P_amb_kpa=101.325):
        """HPR = Q_th / P_el [-]."""
        eta_el = self.eta_electrical(PLR, T_amb_k, P_amb_kpa)
        eta_th = self.eta_thermal(PLR, T_amb_k)
        return np.where(eta_el > 1e-6, eta_th / eta_el, 0.0)

    def heat_rate_kj_kwh(self, PLR, T_amb_k=288.15, P_amb_kpa=101.325):
        """Heat rate [kJ/kWh] -- electrical only."""
        eta_el = self.eta_electrical(PLR, T_amb_k, P_amb_kpa)
        return np.where(eta_el > 1e-6, 3600.0 / eta_el, 0.0)
