"""
EC145 -- Pyrolysis Reactor -- F1b Feedstock Variation Model

Builds on F1a (single-feedstock yield curves) by adding:
  - Feedstock-specific ultimate analysis driving bio-oil/char/gas yields
  - Moisture-LHV coupling:
        LHV_eff = LHV_dry * (1 - M) - h_fg * M
  - Temperature-dependent product distribution (fast vs slow pyrolysis)
  - Part-load thermal efficiency curve: eta(PLR) = a0 + a1*PLR + a2*PLR^2
  - Cellulose/hemicellulose/lignin (CHL) composition effect on bio-oil quality

Product distribution model (empirical, based on Bridgwater 2012):
    bio_oil yield  ~ 0.45 * (1 + CHL_factor) * moisture_factor (fast pyrolysis)
    char yield     ~ 0.15 + 0.30 * lignin_frac
    gas yield      ~ balance (1 - bio_oil - char)
    Temperature effect: bio_oil peaks at ~500 degC; char decreases with T;
                        gas increases with T above 550 degC.

References:
    Bridgwater, A.V. (2012). Review of fast pyrolysis of biomass and product upgrading.
        Biomass & Bioenergy, 38, 68-94.
    Demirbas, A. (2004). Effect of moisture and temperature on pyrolysis product yields.
        Energy Conversion and Management, 45(3), 653-660.
    Oasmaa, A. & Meier, D. (2005). Norms and standards for fast pyrolysis liquids.
        J. Analytical and Applied Pyrolysis, 73(2), 323-334.
    Jenkins, B.M. et al. (1998). Combustion properties of biomass. Fuel Processing Technology,
        54, 17-46.
"""

import numpy as np

_H_FG = 2442.0   # kJ/kg latent heat of vaporisation at 25 degC


class PyrolysisReactorF1b:
    """
    Pyrolysis reactor with feedstock-specific yield, moisture-LHV coupling,
    temperature effect on product distribution, and part-load efficiency.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_rated       = u["Q_rated_kw"]["value"]      # kW thermal input
        self.T_ref         = u["T_ref_degC"]["value"]      # reference T for design yields
        self.PLR_min       = u["PLR_min"]["value"]
        self.a0 = u["a0"]["value"]
        self.a1 = u["a1"]["value"]
        self.a2 = u["a2"]["value"]
        self.T_bio_oil_peak = u["T_bio_oil_peak_degC"]["value"]   # degC
        self.T_gas_rise     = u["T_gas_rise_degC"]["value"]       # degC above which gas increases
        self.feedstock_db   = params["feedstock_db"]

    def _get_feedstock(self, name):
        if name not in self.feedstock_db:
            raise ValueError(
                f"Unknown feedstock: {name}. "
                f"Available: {list(self.feedstock_db.keys())}"
            )
        return self.feedstock_db[name]

    def lhv_effective(self, feedstock_type, moisture_fraction):
        """
        LHV_eff [kJ/kg wet] = LHV_dry * (1 - M) - h_fg * M
        Moisture significantly degrades pyrolysis product quality (Demirbas 2004).
        """
        fs = self._get_feedstock(feedstock_type)
        M = np.clip(float(moisture_fraction), 0.0, 0.60)
        LHV_dry = fs["LHV_dry_MJ_kg"] * 1000.0   # kJ/kg
        LHV_eff = LHV_dry * (1.0 - M) - _H_FG * M
        return max(0.0, LHV_eff), LHV_dry

    def product_yields(self, feedstock_type, temperature_degC, moisture_fraction):
        """
        Bio-oil, char, and gas mass yields [kg/kg dry biomass].

        Temperature effects (Bridgwater 2012):
          - Bio-oil peaks near 500 degC; below 400 or above 600 degC yields fall
          - Char decreases linearly with T (more secondary cracking)
          - Gas is complement

        Moisture effect:
          - Moisture dilutes bio-oil yield (steam consumes energy, reduces organic yield)
          - f_moisture = (1 - 1.2 * M)  [effective yield factor]
        """
        fs = self._get_feedstock(feedstock_type)
        T   = np.clip(float(temperature_degC), 300.0, 700.0)
        M   = np.clip(float(moisture_fraction), 0.0, 0.60)

        # Lignin fraction effect: high lignin -> more char
        lignin = fs.get("lignin_frac", 0.25)

        # Cellulose + hemicellulose fraction: drives bio-oil
        cellulose = fs.get("cellulose_frac", 0.40)
        hemi      = fs.get("hemicellulose_frac", 0.30)
        chl_factor = (cellulose + hemi) - 0.70  # positive if high cellulose

        # Temperature modifier for bio-oil (Gaussian-like peak at T_bio_oil_peak)
        sigma_T = 80.0  # degC width
        f_T_biooil = np.exp(-0.5 * ((T - self.T_bio_oil_peak) / sigma_T) ** 2)

        # Moisture factor: per Demirbas (2004), each 10% moisture drops bio-oil yield ~10%
        f_moisture = max(0.0, 1.0 - 1.2 * M)

        # Base bio-oil yield at peak conditions (fast pyrolysis)
        bio_oil_base = 0.52 + 0.15 * chl_factor   # 0.45-0.55 typical range
        bio_oil = bio_oil_base * f_T_biooil * f_moisture
        bio_oil = float(np.clip(bio_oil, 0.10, 0.70))

        # Char yield: decreases with T (less cracking at low T)
        char_base = 0.22 + 0.20 * lignin       # 0.15-0.35 depending on lignin
        f_T_char  = max(0.3, 1.0 - (T - 300.0) / (700.0 - 300.0) * 0.55)
        char = char_base * f_T_char
        char = float(np.clip(char, 0.05, 0.40))

        # Gas is complement
        gas = max(0.05, 1.0 - bio_oil - char)

        # Re-normalize to 1.0
        total = bio_oil + char + gas
        bio_oil /= total
        char    /= total
        gas     /= total

        return {"bio_oil": float(bio_oil), "char": float(char), "gas": float(gas)}

    def product_lhvs(self, feedstock_type):
        """
        Product LHVs [MJ/kg]:
          - bio_oil: 16-19 MJ/kg (Oasmaa & Meier 2005)
          - char:    28-32 MJ/kg (Jenkins et al. 1998)
          - gas:     variable, ~6-10 MJ/kg (low due to CO2, H2O dilution)
        """
        fs = self._get_feedstock(feedstock_type)
        return {
            "bio_oil": fs.get("LHV_bio_oil_MJ_kg", 17.0),
            "char":    fs.get("LHV_char_MJ_kg",    30.0),
            "gas":     fs.get("LHV_gas_MJ_kg",      7.0),
        }

    def energy_recovery(self, feedstock_type, temperature_degC, moisture_fraction):
        """
        Energy recovery ratio: total product energy / LHV_dry input.
        Accounts for moisture energy penalty.
        """
        yields   = self.product_yields(feedstock_type, temperature_degC, moisture_fraction)
        lhvs     = self.product_lhvs(feedstock_type)
        fs       = self._get_feedstock(feedstock_type)
        LHV_eff, LHV_dry = self.lhv_effective(feedstock_type, moisture_fraction)

        energy_products = (
            yields["bio_oil"] * lhvs["bio_oil"] * 1000.0 +
            yields["char"]    * lhvs["char"]    * 1000.0 +
            yields["gas"]     * lhvs["gas"]     * 1000.0
        )  # kJ/kg dry biomass

        recovery = energy_products / LHV_dry if LHV_dry > 0 else 0.0
        return float(np.clip(recovery, 0.0, 1.05))   # allow slight >1 rounding

    def thermal_efficiency(self, PLR, moisture_fraction, feedstock_type):
        """
        Overall reactor thermal efficiency = (a0 + a1*PLR + a2*PLR^2) * moisture_factor.
        Moisture factor = LHV_eff / LHV_dry.
        """
        PLR_eff = np.clip(np.asarray(PLR, dtype=float), self.PLR_min, 1.0)
        eta_base = self.a0 + self.a1 * PLR_eff + self.a2 * PLR_eff ** 2
        LHV_eff, LHV_dry = self.lhv_effective(feedstock_type, moisture_fraction)
        mf = LHV_eff / LHV_dry if LHV_dry > 0 else 0.0
        return np.clip(eta_base * mf, 0.0, 1.0)

    def predict(self, feedstock_type, temperature_degC, moisture_fraction=0.10,
                PLR=1.0, feed_rate_kg_h=500.0):
        """
        Compute pyrolysis outputs.

        Args:
            feedstock_type:    str from feedstock_db
            temperature_degC:  Reactor temperature [degC] (300-700)
            moisture_fraction: Wet-basis moisture [0-0.60]
            PLR:               Part-load ratio [PLR_min - 1.0]
            feed_rate_kg_h:    Dry biomass feed rate [kg/h]

        Returns:
            dict with:
                bio_oil_yield      : mass fraction [-]
                char_yield         : mass fraction [-]
                gas_yield          : mass fraction [-]
                LHV_eff_MJ_kg      : effective feed LHV [MJ/kg wet]
                moisture_lhv_factor: LHV_eff/LHV_dry [-]
                energy_recovery    : fraction of feed energy in products [-]
                thermal_efficiency : reactor thermal efficiency [-]
                bio_oil_rate_kg_h  : bio-oil production rate [kg/h]
                char_rate_kg_h     : char production rate [kg/h]
                gas_rate_kg_h      : gas production rate [kg/h]
        """
        PLR_eff = float(np.clip(PLR, self.PLR_min, 1.0))
        yields  = self.product_yields(feedstock_type, temperature_degC, moisture_fraction)
        lhvs    = self.product_lhvs(feedstock_type)
        LHV_eff, LHV_dry = self.lhv_effective(feedstock_type, moisture_fraction)
        mf      = LHV_eff / LHV_dry if LHV_dry > 0 else 0.0
        eta_th  = float(self.thermal_efficiency(PLR_eff, moisture_fraction, feedstock_type))
        recovery = self.energy_recovery(feedstock_type, temperature_degC, moisture_fraction)

        m_feed_eff = float(feed_rate_kg_h) * PLR_eff

        return {
            "bio_oil_yield":       yields["bio_oil"],
            "char_yield":          yields["char"],
            "gas_yield":           yields["gas"],
            "LHV_eff_MJ_kg":       LHV_eff / 1000.0,
            "moisture_lhv_factor": mf,
            "energy_recovery":     recovery,
            "thermal_efficiency":  eta_th,
            "bio_oil_rate_kg_h":   m_feed_eff * yields["bio_oil"],
            "char_rate_kg_h":      m_feed_eff * yields["char"],
            "gas_rate_kg_h":       m_feed_eff * yields["gas"],
        }
