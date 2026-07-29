"""
EC141 -- Anaerobic Digester (Thermophilic) -- F1b Feedstock Variation Model

Thermophilic variant (T_opt = 55 degC) of EC140 F1b.

Adds over F1a:
  - Feedstock-specific BMP database (thermophilic BMPs ~5-10% higher than mesophilic)
  - Co-digestion blending and C/N synergy factor
  - C/N ratio yield penalty outside optimal range [20-30]
  - Temperature inhibition model for thermophilic range (45-65 degC)
  - Feedstock-specific hydrolysis rates (thermophilic k ~1.5x mesophilic)
  - Moisture-LHV coupling: LHV_eff = LHV_dry * (1 - M) - h_fg * M
    (same formula as EC087 biomass boiler F1b)

Temperature model (thermophilic):
  - Optimal range: 52-58 degC
  - Below 45 degC: inhibition (transition to mesophilic)
  - Above 60 degC: thermal inhibition of methanogens

References:
    Angelidaki, I. et al. (2009). Water Science & Technology 59(5), 927-934.
    Labatut, R.A. et al. (2011). Bioresource Technology 102, 2606-2614.
    Mata-Alvarez, J. et al. (2014). Renew. Sustain. Energy Rev. 36, 412-427.
"""

import numpy as np

_H_FG = 2442.0   # kJ/kg — latent heat of water vaporisation


class AnaerobicDigesterThermophilicF1b:
    """Thermophilic anaerobic digester with feedstock-specific BMP and co-digestion."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V            = u["digester_volume_m3"]["value"]
        self.k_hydrolysis = u["k_hydrolysis_per_day"]["value"]
        self.T_ref        = u["T_ref_degC"]["value"] + 273.15
        self.E_a          = u["E_a_jmol"]["value"]
        self.R            = u["R_jmolk"]["value"]
        self.LHV_methane  = u["LHV_methane_kwh_m3"]["value"]
        self.T_opt        = u["T_optimum_degC"]["value"]
        self.T_inh_low    = u["T_inhibit_low_degC"]["value"]
        self.T_inh_high   = u["T_inhibit_high_degC"]["value"]
        self.feedstock_db = params["feedstock_db"]
        self.cn_config    = params["cn_ratio"]

    def _parse_feedstock(self, feedstock_type):
        if isinstance(feedstock_type, str):
            if feedstock_type not in self.feedstock_db:
                raise ValueError(f"Unknown feedstock: {feedstock_type}. "
                                 f"Available: {list(self.feedstock_db.keys())}")
            return [(feedstock_type, 1.0, self.feedstock_db[feedstock_type])]
        elif isinstance(feedstock_type, dict):
            total = sum(feedstock_type.values())
            return [(n, f/total, self.feedstock_db[n]) for n, f in feedstock_type.items()
                    if n in self.feedstock_db or (_ for _ in [None] if
                    (_ := None) or True and (n not in self.feedstock_db and
                    (_ := ValueError(f"Unknown feedstock: {n}"))))]
        raise TypeError("feedstock_type must be str or dict")

    def _parse_feedstock(self, feedstock_type):
        if isinstance(feedstock_type, str):
            if feedstock_type not in self.feedstock_db:
                raise ValueError(f"Unknown feedstock: {feedstock_type}. "
                                 f"Available: {list(self.feedstock_db.keys())}")
            return [(feedstock_type, 1.0, self.feedstock_db[feedstock_type])]
        elif isinstance(feedstock_type, dict):
            comps = []
            total = sum(feedstock_type.values())
            for name, frac in feedstock_type.items():
                if name not in self.feedstock_db:
                    raise ValueError(f"Unknown feedstock: {name}")
                comps.append((name, frac/total, self.feedstock_db[name]))
            return comps
        raise TypeError("feedstock_type must be str or dict")

    def blend_properties(self, feedstock_type):
        components = self._parse_feedstock(feedstock_type)
        BMP = C = N = ch4 = k = 0.0
        for name, frac, data in components:
            BMP += frac * data["BMP_L_CH4_kgVS"]
            C   += frac * data["C_pct"]
            N   += frac * data["N_pct"]
            ch4 += frac * data["methane_content"]
            k   += frac * data["k_hydrolysis"]
        cn = C / N if N > 0 else 100.0
        return {"BMP": BMP, "C_pct": C, "N_pct": N, "cn_ratio": cn,
                "methane_content": ch4, "k_hydrolysis": k, "n_components": len(components)}

    def temperature_factor(self, temp_c):
        """
        Thermophilic temperature factor.
        Optimal at 52-58 degC. Inhibited below 45 degC and above 62 degC.
        Uses Arrhenius within the optimal window.
        """
        T = float(temp_c) + 273.15
        f_T = np.exp(-self.E_a / self.R * (1.0 / T - 1.0 / self.T_ref))

        # Inhibition below thermophilic threshold
        if temp_c < self.T_inh_low:
            # Linear decay from T_inh_low down to 35 degC (mesophilic starts)
            f_T = f_T * max(0.0, (temp_c - 35.0) / (self.T_inh_low - 35.0))
        # Inhibition above upper limit
        elif temp_c > self.T_inh_high:
            f_T = f_T * max(0.0, (70.0 - temp_c) / (70.0 - self.T_inh_high))

        return max(0.0, float(f_T))

    def co_digestion_synergy(self, cn_ratio, n_components):
        if n_components <= 1:
            return 1.0
        cn_opt = 25.0
        cn_min = self.cn_config["optimal_min"]
        cn_max = self.cn_config["optimal_max"]
        if cn_min <= cn_ratio <= cn_max:
            prox = 1.0 - abs(cn_ratio - cn_opt) / (cn_max - cn_min)
            return 1.0 + 0.10 * prox
        return 1.0

    def cn_ratio_penalty(self, cn_ratio):
        cn_min = self.cn_config["optimal_min"]
        cn_max = self.cn_config["optimal_max"]
        pf = self.cn_config["penalty_factor"]
        if cn_min <= cn_ratio <= cn_max:
            return 1.0
        dev = (cn_min - cn_ratio) if cn_ratio < cn_min else (cn_ratio - cn_max)
        return max(0.3, 1.0 - pf * dev)

    def vs_removal_efficiency(self, hrt_days, k_hydrolysis):
        return 1.0 - np.exp(-float(k_hydrolysis) * float(hrt_days))

    def moisture_lhv_factor(self, moisture_fraction):
        """
        LHV correction for wet feedstock.
        LHV_eff = LHV_dry * (1 - M) - h_fg * M
        Applied as a multiplier on BMP (energy extraction limited by available LHV).
        (EC087 biomass boiler F1b convention; Jenkins et al. 1998)
        """
        M = np.clip(float(moisture_fraction), 0.0, 0.95)
        LHV_dry = 18000.0  # kJ/kg (typical biomass HHV ~19 MJ/kg)
        LHV_eff = LHV_dry * (1.0 - M) - _H_FG * M
        return max(0.0, LHV_eff / LHV_dry)

    def predict(self, feedstock_type, vs_loading_kg_m3_day,
                temperature_degC=55.0, hrt_days=15.0, moisture_fraction=0.0):
        """
        Predict biogas production for thermophilic digester.

        Args:
            feedstock_type:        str or dict of blends
            vs_loading_kg_m3_day:  VS loading [kgVS/(m3*day)]
            temperature_degC:      operating temperature [degC]
            hrt_days:              hydraulic retention time [days]
            moisture_fraction:     wet-basis moisture fraction [0-1]

        Returns:
            dict: biogas_yield_m3_day, methane_content_pct, methane_yield_m3_day,
                  vs_removal_pct, cn_ratio, moisture_lhv_factor
        """
        blend = self.blend_properties(feedstock_type)
        f_T = self.temperature_factor(temperature_degC)
        k_eff = blend["k_hydrolysis"]
        yield_factor = 1.0 - np.exp(-k_eff * float(hrt_days))
        cn_penalty = self.cn_ratio_penalty(blend["cn_ratio"])
        synergy = self.co_digestion_synergy(blend["cn_ratio"], blend["n_components"])
        f_moisture = self.moisture_lhv_factor(moisture_fraction)

        BMP_eff = blend["BMP"] * yield_factor * f_T * cn_penalty * synergy * f_moisture
        VS_input = float(vs_loading_kg_m3_day) * self.V
        methane_yield = BMP_eff / 1000.0 * VS_input
        methane_content = blend["methane_content"]
        biogas_yield = methane_yield / methane_content if methane_content > 0 else 0.0
        vs_removal = self.vs_removal_efficiency(hrt_days, k_eff) * 100.0

        return {
            "biogas_yield_m3_day":  float(biogas_yield),
            "methane_content_pct":  float(methane_content * 100.0),
            "methane_yield_m3_day": float(methane_yield),
            "vs_removal_pct":       float(vs_removal),
            "cn_ratio":             float(blend["cn_ratio"]),
            "moisture_lhv_factor":  float(f_moisture),
        }
