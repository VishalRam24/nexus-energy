"""
EC154 -- Enhanced Geothermal System (EGS) -- F1b Reservoir Decline & ORC Part-Load Model

Builds on F1a (exergy + pump parasitic) by adding:
  - Reservoir thermal breakthrough and progressive temperature decline
  - Fracture permeability degradation over time (pressure-dependent closing)
  - ORC part-load efficiency curve (same as binary, slower efficiency drop)
  - Condenser ambient sensitivity (air-cooled ORC typical for EGS)
  - Pump parasitic scaling with reservoir pressure decline

EGS Reservoir Thermal Decline:
    EGS heat extraction from hot dry rock (HDR). As cold water circulates,
    the rock near fractures cools — "thermal breakthrough".
    Model: T_out(t) = T_geo - (T_geo - T_inj) * (1 - exp(-t/tau_th))
    where tau_th = thermal time constant [years].
    Tester et al. (2006): tau_th ~ 5-20 years depending on fracture spacing.

Fracture Permeability Decline:
    In-situ stress closes fractures under production-induced pressure changes.
    Effective permeability k_eff = k_0 * exp(-c_perm * delta_P)
    Flow rate proportional to k_eff: m_dot_eff = m_dot_0 * (k_eff / k_0)^0.5
    Tester et al. (2006) MIT EGS report; Watanabe et al. (2017).

Pump Parasitic Scaling:
    As permeability declines, pump power must increase to maintain flow.
    pump_parasitic_effective = min(pump_base / sqrt(k_ratio), 0.5)
    Limits net power — critical design constraint for EGS.

References:
    Tester, J.W. et al. (2006). The Future of Geothermal Energy, MIT/DOE.
    DiPippo, R. (2015). Geothermal Power Plants, 4th ed., Chapter 16.
    Watanabe, N. et al. (2017). Hydraulic stimulation of the Pohang EGS.
        Geothermics, 68, 115-122.
    Sanyal, S.K. & Butler, S.J. (2005). An analysis of power generation prospects
        from Enhanced Geothermal Systems. GRC Transactions, 29.
"""

import numpy as np


class EGSF1b:
    """
    EGS with reservoir thermal breakthrough, fracture permeability decline,
    ORC part-load, and ambient derating.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.cp_fluid         = u["cp_fluid_J_kgK"]["value"]           # J/(kg*K)
        self.eta_util         = u["eta_utilization"]["value"]           # -
        self.pump_parasitic_0 = u["pump_parasitic_fraction_0"]["value"] # fraction at t=0
        self.T_geo_design     = u["T_geo_design_degC"]["value"]         # degC
        self.T_reject_design  = u["T_reject_design_degC"]["value"]      # degC
        self.T_reinject_offset= u["T_reinject_offset"]["value"]         # degC above T_reject
        self.P_rated          = u["P_rated_kw"]["value"]                # kW
        self.m_dot_design     = u["m_dot_design_kg_s"]["value"]         # kg/s
        self.tau_thermal      = u["tau_thermal_yr"]["value"]            # thermal time constant [yr]
        self.T_inject         = u["T_inject_degC"]["value"]             # injection T [degC]
        self.c_perm           = u["c_perm_decline"]["value"]            # permeability decline coeff
        self.delta_P_design   = u["delta_P_design_MPa"]["value"]        # design pressure diff [MPa]
        self.k_amb            = u["ambient_derating_coeff"]["value"]    # 1/degC
        self.PLR_min          = u["PLR_min"]["value"]
        self.pump_max_frac    = u["pump_parasitic_max"]["value"]        # cap on pump fraction

        plc = params["part_load_curve"]
        self.plr_points    = np.array(plc["PLR_points"])
        self.eta_ratio_pts = np.array(plc["eta_ratio"])

    # ------------------------------------------------------------------
    # Reservoir thermal breakthrough
    # ------------------------------------------------------------------
    def reservoir_temperature(self, T_geo_c: float, years_operation: float) -> float:
        """
        Effective wellhead fluid temperature after thermal breakthrough.
        T_out(t) = T_geo - (T_geo - T_inj) * (1 - exp(-t/tau_th))

        At t=0: T_out = T_geo (no breakthrough)
        At t >> tau_th: T_out -> T_inj (complete breakthrough)

        Tester et al. (2006): tau_th = 5-20 yr for typical fracture spacing.
        """
        T_geo  = float(T_geo_c)
        T_inj  = float(self.T_inject)
        tau    = max(0.1, float(self.tau_thermal))
        t      = max(0.0, float(years_operation))
        T_out  = T_geo - (T_geo - T_inj) * (1.0 - np.exp(-t / tau))
        return float(T_out)

    # ------------------------------------------------------------------
    # Fracture permeability & flow rate decline
    # ------------------------------------------------------------------
    def permeability_ratio(self, years_operation: float,
                            delta_P_MPa: float = None) -> float:
        """
        Permeability ratio k_eff/k_0 due to stress-induced fracture closure.
        k_ratio = exp(-c_perm * delta_P * (t / t_ref))
        where t_ref = 1 year (normalisation).
        Bounded: [0.1, 1.0] — fractures retain at least 10% permeability.
        """
        t = max(0.0, float(years_operation))
        dP = float(delta_P_MPa) if delta_P_MPa is not None else self.delta_P_design
        k_ratio = np.exp(-self.c_perm * dP * t)
        return float(np.clip(k_ratio, 0.1, 1.0))

    def effective_flow_rate(self, m_dot_0: float, years_operation: float,
                             delta_P_MPa: float = None) -> float:
        """
        Effective circulation flow rate after permeability decline.
        m_dot_eff = m_dot_0 * sqrt(k_ratio)   [Darcy-law scaling]
        """
        k_ratio = self.permeability_ratio(years_operation, delta_P_MPa)
        return float(m_dot_0 * np.sqrt(k_ratio))

    # ------------------------------------------------------------------
    # Pump parasitic (increases as permeability drops)
    # ------------------------------------------------------------------
    def pump_parasitic_effective(self, years_operation: float,
                                  delta_P_MPa: float = None) -> float:
        """
        Effective pump parasitic fraction. As k decreases, pump must work harder.
        pump_eff = pump_0 / sqrt(k_ratio)   (flow vs pressure scaling)
        Capped at pump_max_frac to reflect cavitation limits.
        """
        k_ratio = self.permeability_ratio(years_operation, delta_P_MPa)
        pump_eff = self.pump_parasitic_0 / max(np.sqrt(k_ratio), 0.01)
        return float(np.clip(pump_eff, 0.0, self.pump_max_frac))

    # ------------------------------------------------------------------
    # Part-load
    # ------------------------------------------------------------------
    def f_plr(self, PLR: float) -> float:
        PLR_c = np.clip(float(PLR), self.PLR_min, 1.0)
        return float(np.interp(PLR_c, self.plr_points, self.eta_ratio_pts))

    # ------------------------------------------------------------------
    # Ambient derating (air-cooled ORC)
    # ------------------------------------------------------------------
    def condenser_factor(self, T_reject_degC: float) -> float:
        """
        Air-cooled ORC condenser derating (typical for remote EGS sites).
        f = 1 - k_amb * max(0, T_rej - T_rej_design)
        """
        dT = max(0.0, float(T_reject_degC) - self.T_reject_design)
        f  = 1.0 - self.k_amb * dT
        if float(T_reject_degC) < self.T_reject_design:
            f = 1.0 + 0.002 * (self.T_reject_design - float(T_reject_degC))
        return float(np.clip(f, 0.5, 1.15))

    # ------------------------------------------------------------------
    # Main predict
    # ------------------------------------------------------------------
    def predict(self, T_geo_degC: float, m_dot_kg_s: float,
                T_reject_degC: float, PLR: float = 1.0,
                years_operation: float = 0.0,
                delta_P_MPa: float = None) -> dict:
        """
        Compute EGS performance with thermal breakthrough, permeability decline,
        ORC part-load, and ambient derating.

        Args:
            T_geo_degC:      Initial rock/fluid temperature [degC]
            m_dot_kg_s:      Design circulation flow rate [kg/s]
            T_reject_degC:   Ambient rejection temperature [degC]
            PLR:             Part-load ratio [PLR_min - 1.0]
            years_operation: Years since start of production [years]
            delta_P_MPa:     Reservoir pressure differential [MPa]; uses design if None

        Returns:
            dict with:
                power_output_kw       : Net electrical output [kW]
                gross_efficiency      : Gross cycle efficiency [-]
                net_efficiency        : Net efficiency after pump parasitics [-]
                resource_factor       : Reservoir temperature retention factor [-]
                permeability_ratio    : k_eff/k_0 [-]
                pump_parasitic_frac   : Effective pump parasitic fraction [-]
                effective_flow_kg_s   : Actual flow rate after perm decline [kg/s]
                T_out_degC            : Effective wellhead temperature [degC]
        """
        PLR   = np.clip(float(PLR), self.PLR_min, 1.0)
        T_geo = float(T_geo_degC)
        m_0   = float(m_dot_kg_s)
        T_rej = float(T_reject_degC)
        years = max(0.0, float(years_operation))
        dP    = float(delta_P_MPa) if delta_P_MPa is not None else self.delta_P_design

        # Reservoir effects
        T_out    = self.reservoir_temperature(T_geo, years)
        m_eff    = self.effective_flow_rate(m_0, years, dP)
        k_ratio  = self.permeability_ratio(years, dP)
        pump_frac = self.pump_parasitic_effective(years, dP)

        # Resource temperature retention factor (relative to design)
        f_res = (T_out - self.T_inject) / (T_geo - self.T_inject) if (T_geo - self.T_inject) > 0 else 0.0
        f_res = float(np.clip(f_res, 0.0, 1.0))

        # Derating factors
        f_pl   = self.f_plr(PLR)
        f_cond = self.condenser_factor(T_rej)

        # Reinjection temperature
        T_reinj = T_rej + self.T_reinject_offset

        # Heat extraction
        dT   = max(0.0, T_out - T_reinj)
        Q_in = m_eff * self.cp_fluid * dT / 1000.0  # kW

        # Carnot efficiency
        T_out_K   = T_out  + 273.15
        T_reinj_K = T_reinj + 273.15
        eta_carnot = max(0.0, 1.0 - T_reinj_K / T_out_K)

        # Gross efficiency
        eta_gross = float(np.clip(self.eta_util * eta_carnot * f_pl * f_cond, 0.0, 0.25))

        # Net efficiency
        eta_net = float(np.clip(eta_gross * (1.0 - pump_frac), 0.0, 0.20))

        # Power
        P_gross  = Q_in * eta_gross
        P_net    = P_gross * (1.0 - pump_frac)
        P_output = min(max(0.0, P_net * PLR), self.P_rated)

        return {
            "power_output_kw":     float(P_output),
            "gross_efficiency":    eta_gross,
            "net_efficiency":      eta_net,
            "resource_factor":     f_res,
            "permeability_ratio":  float(k_ratio),
            "pump_parasitic_frac": float(pump_frac),
            "effective_flow_kg_s": float(m_eff),
            "T_out_degC":          float(T_out),
        }
