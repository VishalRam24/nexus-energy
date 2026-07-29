"""
EC152 -- Flash Steam Geothermal Plant -- F1b Part-Load & Ambient Derating Model

Builds on F1a (flash pressure optimisation) by adding:
  - Part-load turbine efficiency curve (different from dry steam due to flash stage)
  - Brine chemistry / scaling index effect on flash efficiency and HX fouling
  - Condenser T sensitivity to ambient (air-cooled or wet-tower)
  - Reservoir brine temperature decline over plant lifetime (1-2%/yr)
  - Double-flash bonus at high brine temperatures (>200 degC)

Brine chemistry (scaling):
    Geothermal brines contain silica, calcite, sulphates that scale flash vessels/HX.
    Langelier Saturation Index (LSI) proxy: function of TDS and temperature.
    High LSI -> more scaling -> lower effective heat transfer -> reduced Q and efficiency.
    Corrosion/scaling can reduce output by 5-15% without mitigation (Vaca-Mier et al. 2003).

    f_scale = 1 - k_scale * max(0, LSI - LSI_ref)
    LSI simplified: LSI_proxy = TDS_g_L * T_brine / T_ref / 1000

Double-flash option:
    When T_brine > T_double_flash_min, a second flash stage captures additional steam.
    Approximate benefit: +15-25% power over single flash.
    T_flash2_opt = sqrt(T_flash1_K * T_cond_K) - 273.15 (DiPippo 2015, Chapter 6)

References:
    DiPippo, R. (2015). Geothermal Power Plants, 4th ed. Butterworth-Heinemann.
        Chapter 6 — Double-Flash Steam Plants.
    Zarrouk, S.J. & Moon, H. (2014). Efficiency of geothermal power plants: A
        worldwide review. Geothermics, 51, 142-153.
    Vaca-Mier, M. et al. (2003). Effect of silica scaling in geothermal power plants.
        Geothermics, 32(4-6), 603-612.
    Lukawski, M.Z. et al. (2014). PROCEEDINGS, 39th Workshop on Geothermal
        Reservoir Engineering, Stanford University.
"""

import numpy as np


class FlashSteamGeothermalF1b:
    """
    Flash steam geothermal plant with part-load, ambient derating,
    brine chemistry scaling, and reservoir decline.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.cp_brine           = u["cp_brine_J_kgK"]["value"]           # J/(kg*K)
        self.eta_util           = u["eta_utilization"]["value"]           # -
        self.T_cond_offset      = u["T_condenser_offset"]["value"]        # degC
        self.P_rated            = u["P_rated_kw"]["value"]                # kW
        self.T_brine_design     = u["T_brine_design_degC"]["value"]       # degC
        self.T_reject_design    = u["T_reject_design_degC"]["value"]      # degC
        self.m_dot_design       = u["m_dot_brine_design_kg_s"]["value"]   # kg/s
        self.decline_rate       = u["decline_rate_per_yr"]["value"]       # 1/yr
        self.k_amb              = u["ambient_derating_coeff"]["value"]    # 1/degC
        self.cooling_mode       = u["cooling_mode"]["value"]              # "wet"/"dry"
        self.k_scale            = u["scale_penalty_per_lsi"]["value"]     # fractional/LSI unit
        self.LSI_ref            = u["LSI_reference"]["value"]             # reference LSI
        self.TDS_base           = u["TDS_base_g_L"]["value"]              # g/L TDS at design
        self.T_double_flash_min = u["T_double_flash_min_degC"]["value"]   # degC
        self.double_flash_bonus = u["double_flash_bonus"]["value"]        # fraction (0.0-0.25)
        self.PLR_min            = u["PLR_min"]["value"]

        plc = params["part_load_curve"]
        self.plr_points    = np.array(plc["PLR_points"])
        self.eta_ratio_pts = np.array(plc["eta_ratio"])

    # ------------------------------------------------------------------
    # Steam table helper
    # ------------------------------------------------------------------
    @staticmethod
    def h_fg(T_c):
        """Latent heat of vaporisation at T_c [degC] -> kJ/kg. Linear steam table fit."""
        T = np.asarray(T_c, dtype=float)
        return np.clip(2501.0 - 2.361 * T, 100.0, None)

    def optimal_flash_temperature(self, T_geo_c, T_reject_c):
        """Single-flash optimal temperature [degC] = geometric mean of T_geo and T_cond (K)."""
        T_geo  = float(T_geo_c)  + 273.15
        T_cond = float(T_reject_c) + self.T_cond_offset + 273.15
        return float(np.sqrt(T_geo * T_cond) - 273.15)

    def steam_quality(self, T_geo_c, T_flash_c):
        """Dryness fraction x = cp*(T_geo - T_flash) / h_fg(T_flash)."""
        dT   = max(0.0, float(T_geo_c) - float(T_flash_c))
        hfg  = float(self.h_fg(T_flash_c)) * 1000.0  # kJ->J
        return float(np.clip(self.cp_brine * dT / hfg, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Part-load
    # ------------------------------------------------------------------
    def f_plr(self, PLR: float) -> float:
        """Part-load efficiency ratio (flash turbine has similar characteristic to ORC)."""
        PLR_c = np.clip(float(PLR), self.PLR_min, 1.0)
        return float(np.interp(PLR_c, self.plr_points, self.eta_ratio_pts))

    # ------------------------------------------------------------------
    # Condenser / ambient
    # ------------------------------------------------------------------
    def condenser_factor(self, T_reject_degC: float) -> float:
        """
        Carnot-ratio ambient derating.
        Wet tower: dT_cond = 0.5 * dT_amb; air: dT_cond = 1.0 * dT_amb.
        """
        dT_amb = float(T_reject_degC) - self.T_reject_design
        coupling = 0.5 if self.cooling_mode == "wet" else 1.0

        T_hot   = self.T_brine_design + 273.15
        T_cond_d = (self.T_reject_design + self.T_cond_offset) + 273.15
        T_cond_a = T_cond_d + coupling * dT_amb

        eta_d = 1.0 - T_cond_d / T_hot
        eta_a = 1.0 - T_cond_a / T_hot
        f = (eta_a / eta_d) if eta_d > 0 else 1.0
        return float(np.clip(f, 0.5, 1.2))

    # ------------------------------------------------------------------
    # Reservoir decline
    # ------------------------------------------------------------------
    def resource_factor(self, years_operation: float) -> float:
        """Brine temperature / flow decline: f = (1 - decline_rate)^years."""
        years = max(0.0, float(years_operation))
        return float((1.0 - self.decline_rate) ** years)

    # ------------------------------------------------------------------
    # Brine chemistry / scaling
    # ------------------------------------------------------------------
    def scaling_factor(self, TDS_g_L: float, T_brine_degC: float) -> float:
        """
        Scaling penalty factor based on simplified Langelier Saturation Index proxy.
        LSI_proxy = TDS_g_L * T_brine / (TDS_base * T_brine_design)
        f_scale = 1 - k_scale * max(0, LSI_proxy - LSI_ref)
        Bounded: [0.7, 1.0] — severe scaling reduces output up to 30%.

        Vaca-Mier et al. (2003): silica scaling 5-15% output reduction;
        CaCO3 scaling can be more severe in high-TDS fields.
        """
        lsi = float(TDS_g_L) * float(T_brine_degC) / (self.TDS_base * self.T_brine_design)
        f = 1.0 - self.k_scale * max(0.0, lsi - self.LSI_ref)
        return float(np.clip(f, 0.7, 1.0))

    # ------------------------------------------------------------------
    # Double-flash bonus
    # ------------------------------------------------------------------
    def flash_config_factor(self, T_brine_degC: float) -> float:
        """
        Returns power multiplier for double-flash when T_brine exceeds threshold.
        Single flash: factor = 1.0
        Double flash: factor = 1 + double_flash_bonus (typically 0.15-0.25)
        DiPippo (2015) Chapter 6: double flash adds ~25% at 240 degC.
        """
        if float(T_brine_degC) >= self.T_double_flash_min:
            return 1.0 + self.double_flash_bonus
        return 1.0

    # ------------------------------------------------------------------
    # Main predict
    # ------------------------------------------------------------------
    def predict(self, T_brine_degC: float, m_dot_brine_kg_s: float,
                T_reject_degC: float, PLR: float = 1.0,
                years_operation: float = 0.0,
                TDS_g_L: float = None) -> dict:
        """
        Compute flash steam plant performance with all derating factors.

        Args:
            T_brine_degC:      Wellhead brine temperature [degC]
            m_dot_brine_kg_s:  Brine mass flow rate [kg/s]
            T_reject_degC:     Cooling rejection temperature [degC]
            PLR:               Part-load ratio [PLR_min - 1.0]
            years_operation:   Years since commissioning [years]
            TDS_g_L:           Total dissolved solids [g/L]; uses base if None

        Returns:
            dict with:
                power_output_kw   : Net electrical output [kW]
                efficiency        : Overall net efficiency [-]
                resource_factor   : Brine resource decline factor [-]
                condenser_factor  : Ambient derating factor [-]
                scaling_factor    : Brine chemistry scaling factor [-]
                flash_config      : Single (1.0) or double-flash multiplier [-]
                steam_quality     : Dryness fraction at flash point [-]
        """
        PLR    = np.clip(float(PLR), self.PLR_min, 1.0)
        T_b    = float(T_brine_degC)
        m_dot  = float(m_dot_brine_kg_s)
        T_rej  = float(T_reject_degC)
        years  = max(0.0, float(years_operation))
        tds    = float(TDS_g_L) if TDS_g_L is not None else self.TDS_base

        # Derating factors
        f_pl    = self.f_plr(PLR)
        f_cond  = self.condenser_factor(T_rej)
        f_res   = self.resource_factor(years)
        f_scale = self.scaling_factor(tds, T_b)
        f_flash = self.flash_config_factor(T_b)

        # Effective brine temperature after resource decline
        T_ground = 15.0
        T_b_eff  = T_ground + (T_b - T_ground) * f_res

        # Optimal flash temperature
        T_flash  = self.optimal_flash_temperature(T_b_eff, T_rej)

        # Steam quality
        x = self.steam_quality(T_b_eff, T_flash)

        # Condenser temperature
        T_cond = T_rej + self.T_cond_offset

        # Heat from steam to condenser
        dT_flash = max(0.0, T_b_eff - T_flash)
        Q_in = m_dot * self.cp_brine * dT_flash / 1000.0  # kW (pre-flash heat)

        # Carnot efficiency
        T_b_K    = T_b_eff + 273.15
        T_cond_K = T_cond  + 273.15
        eta_carnot = max(0.0, 1.0 - T_cond_K / T_b_K)

        # Effective efficiency (utilisation * Carnot * part-load * ambient * scaling * flash config)
        eta_eff = self.eta_util * eta_carnot * f_pl * f_cond * f_scale * f_flash
        eta_eff = float(np.clip(eta_eff, 0.0, 0.30))

        # Power
        P_gross  = Q_in * eta_eff
        P_output = min(P_gross * PLR, self.P_rated)
        P_output = max(0.0, float(P_output))

        return {
            "power_output_kw":  P_output,
            "efficiency":       eta_eff,
            "resource_factor":  float(f_res),
            "condenser_factor": float(f_cond),
            "scaling_factor":   float(f_scale),
            "flash_config":     float(f_flash),
            "steam_quality":    float(x),
        }
