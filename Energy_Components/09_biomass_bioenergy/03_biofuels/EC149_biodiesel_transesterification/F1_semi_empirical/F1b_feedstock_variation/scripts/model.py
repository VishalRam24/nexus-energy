"""
EC149 -- Biodiesel Transesterification -- F1b Feedstock Variation Model

Builds on F1a (single-feedstock yield) by adding:
  - Feedstock-specific oil content, fatty acid profile, and FFA content
  - Moisture-LHV coupling:
        LHV_eff = LHV_dry * (1 - M) - h_fg * M
  - Temperature effect on reaction rate (Arrhenius, optimal 55-65 degC for alkali)
  - Free fatty acid (FFA) penalty: high FFA -> saponification -> reduced yield
  - Part-load efficiency curve
  - Co-product glycerol and methanol recovery

Transesterification stoichiometry (base-catalyzed, Freedman 1984):
    Triglyceride + 3 CH3OH -> 3 FAME + Glycerol
    Molar ratio: 1 mol TG + 3 mol MeOH
    Mass yield: ~1.02 kg FAME / kg triglyceride (slightly > 1 due to MeOH addition)

FFA penalty (Rashid 2008):
    FFA > 1%: saponification competes, each 1% FFA reduces yield ~2%
    FFA > 5%: pre-esterification step needed

References:
    Freedman, B. et al. (1984). Variables affecting the transesterification.
        J. American Oil Chemists' Society, 61(10), 1638-1643.
    Rashid, U. & Anwar, F. (2008). Production of biodiesel from high FFA sunflower oil.
        Bioresource Technology, 99, 6624-6629.
    Meher, L.C. et al. (2006). Technical aspects of biodiesel production.
        Renewable & Sustainable Energy Reviews, 10, 248-268.
    Ma, F. & Hanna, M.A. (1999). Biodiesel production: a review.
        Bioresource Technology, 70, 1-15.
"""

import numpy as np

_H_FG = 2442.0       # kJ/kg latent heat
_E_A  = 45000.0      # J/mol alkali transesterification activation energy
_R    = 8.314        # J/(mol*K)
_T_OPT = 60.0        # degC optimal temperature


class BiodieselTransesterificationF1b:
    """
    Biodiesel transesterification with feedstock-specific oil content, FFA content,
    temperature kinetics, moisture-LHV coupling, and part-load efficiency.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.PLR_min       = u["PLR_min"]["value"]
        self.a0 = u["a0"]["value"]
        self.a1 = u["a1"]["value"]
        self.a2 = u["a2"]["value"]
        self.MeOH_ratio    = u["methanol_to_oil_molar_ratio"]["value"]   # typically 6:1
        self.base_yield    = u["base_conversion"]["value"]               # max conversion
        self.ffa_penalty   = u["ffa_penalty_per_pct"]["value"]           # per %FFA
        self.feedstock_db  = params["feedstock_db"]

    def _get_feedstock(self, name):
        if name not in self.feedstock_db:
            raise ValueError(
                f"Unknown feedstock: {name}. "
                f"Available: {list(self.feedstock_db.keys())}"
            )
        return self.feedstock_db[name]

    def lhv_effective(self, feedstock_type, moisture_fraction):
        """LHV_eff [kJ/kg wet] = LHV_dry*(1-M) - h_fg*M."""
        fs = self._get_feedstock(feedstock_type)
        M  = np.clip(float(moisture_fraction), 0.0, 0.30)
        LHV_dry = fs["LHV_dry_MJ_kg"] * 1000.0
        LHV_eff = LHV_dry * (1.0 - M) - _H_FG * M
        return max(0.0, LHV_eff), LHV_dry

    def temperature_factor(self, temperature_degC):
        """
        Arrhenius reaction rate factor normalized to optimal T.
        Below T_opt: activation increases rate.
        Above T_opt: methanol volatilization and side reactions reduce yield.
        """
        T    = float(temperature_degC)
        T_K  = T + 273.15
        T_opt_K = _T_OPT + 273.15

        if T <= _T_OPT:
            f = np.exp(-_E_A / _R * (1.0 / T_K - 1.0 / T_opt_K))
            f = max(0.10, float(f))
        else:
            # Above optimal: MeOH vaporizes, reduces conversion
            k_decline = 0.012   # 1/degC
            f = np.exp(-k_decline * (T - _T_OPT))
            f = max(0.05, float(f))

        return float(np.clip(f, 0.0, 1.0))

    def ffa_factor(self, ffa_pct):
        """
        FFA (free fatty acid) conversion penalty.
        Each 1% FFA reduces yield by ffa_penalty (Rashid 2008).
        Below 1% FFA: negligible saponification.
        """
        ffa = max(0.0, float(ffa_pct))
        if ffa <= 1.0:
            return 1.0
        penalty = self.ffa_penalty * (ffa - 1.0)
        return float(max(0.1, 1.0 - penalty))

    def biodiesel_yield(self, feedstock_type, temperature_degC, ffa_override=None):
        """
        Biodiesel mass yield [kg FAME / kg raw oil feedstock].
        Y_FAME = oil_content * base_yield * f_T * f_FFA * FAME_stoich_factor
        FAME stoichiometric factor: ~1.02 (MeOH group replaces glycerol, slight mass gain)
        """
        fs   = self._get_feedstock(feedstock_type)
        oil  = fs["oil_content_frac"]
        ffa  = ffa_override if ffa_override is not None else fs.get("ffa_pct", 0.5)

        f_T   = self.temperature_factor(temperature_degC)
        f_ffa = self.ffa_factor(ffa)

        Y_FAME = oil * self.base_yield * f_T * f_ffa * 1.02
        return float(np.clip(Y_FAME, 0.0, 0.95))

    def glycerol_yield(self, feedstock_type, temperature_degC, ffa_override=None):
        """
        Glycerol co-product yield [kg / kg raw feedstock].
        ~10% of FAME yield on mass basis (from triglyceride backbone).
        """
        Y_FAME = self.biodiesel_yield(feedstock_type, temperature_degC, ffa_override)
        return float(Y_FAME * 0.10)

    def thermal_efficiency(self, PLR, moisture_fraction, feedstock_type):
        """Process thermal efficiency with part-load polynomial and moisture factor."""
        PLR_eff  = np.clip(np.asarray(PLR, dtype=float), self.PLR_min, 1.0)
        eta_base = self.a0 + self.a1 * PLR_eff + self.a2 * PLR_eff ** 2
        LHV_eff, LHV_dry = self.lhv_effective(feedstock_type, moisture_fraction)
        mf = LHV_eff / LHV_dry if LHV_dry > 0 else 0.0
        return float(np.clip(eta_base * mf, 0.0, 1.0))

    def predict(self, feedstock_type, temperature_degC=60.0, moisture_fraction=0.05,
                PLR=1.0, feed_rate_kg_h=1000.0, ffa_override=None):
        """
        Compute biodiesel production outputs.

        Args:
            feedstock_type:    str from feedstock_db
            temperature_degC:  Reaction temperature [degC] (40-80)
            moisture_fraction: Feed wet-basis moisture [0-0.30]
            PLR:               Part-load ratio [PLR_min-1.0]
            feed_rate_kg_h:    Raw feedstock feed rate [kg/h]
            ffa_override:      Optional FFA % to override feedstock default

        Returns:
            dict with:
                biodiesel_yield    : kg FAME/kg_raw_feed [-]
                glycerol_yield     : kg glycerol/kg_raw_feed [-]
                oil_content        : oil fraction of raw feed [-]
                temperature_factor : reaction rate factor [-]
                ffa_penalty_factor : FFA conversion penalty factor [-]
                LHV_eff_MJ_kg      : effective feed LHV [MJ/kg wet]
                moisture_lhv_factor: LHV_eff/LHV_dry [-]
                thermal_efficiency : process efficiency [-]
                biodiesel_rate_kg_h: FAME production [kg/h]
                glycerol_rate_kg_h : glycerol production [kg/h]
        """
        PLR_eff = float(np.clip(PLR, self.PLR_min, 1.0))
        fs      = self._get_feedstock(feedstock_type)
        ffa     = ffa_override if ffa_override is not None else fs.get("ffa_pct", 0.5)

        Y_FAME = self.biodiesel_yield(feedstock_type, temperature_degC, ffa)
        Y_glyc = self.glycerol_yield(feedstock_type, temperature_degC, ffa)
        f_T    = self.temperature_factor(temperature_degC)
        f_ffa  = self.ffa_factor(ffa)
        LHV_eff, LHV_dry = self.lhv_effective(feedstock_type, moisture_fraction)
        mf     = LHV_eff / LHV_dry if LHV_dry > 0 else 0.0
        eta_th = float(self.thermal_efficiency(PLR_eff, moisture_fraction, feedstock_type))

        m_feed = float(feed_rate_kg_h) * PLR_eff

        return {
            "biodiesel_yield":     Y_FAME,
            "glycerol_yield":      Y_glyc,
            "oil_content":         float(fs["oil_content_frac"]),
            "temperature_factor":  f_T,
            "ffa_penalty_factor":  f_ffa,
            "LHV_eff_MJ_kg":       LHV_eff / 1000.0,
            "moisture_lhv_factor": mf,
            "thermal_efficiency":  eta_th,
            "biodiesel_rate_kg_h": m_feed * Y_FAME,
            "glycerol_rate_kg_h":  m_feed * Y_glyc,
        }
