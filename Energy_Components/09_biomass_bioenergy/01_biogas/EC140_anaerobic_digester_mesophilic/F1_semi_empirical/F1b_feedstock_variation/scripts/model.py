"""
EC140 -- Anaerobic Digester (Mesophilic) -- F1b Feedstock Variation Model

Builds on F1a (single-feedstock yield) by adding:
  - Feedstock-specific Biochemical Methane Potential (BMP) database
  - Co-digestion blending with synergy factor
  - C/N ratio calculation and yield penalty outside optimal range [20-30]
  - Feedstock-specific hydrolysis rates and methane content
  - VS removal efficiency estimation

BMP database (L CH4/kgVS at STP):
    cattle_manure:  250
    food_waste:     400
    grass_silage:   350
    sewage_sludge:  200
    corn_silage:    340

Co-digestion synergy: Up to 10% bonus when blending carbon-rich and nitrogen-rich
feedstocks (C/N ratio approaches optimal 20-30).

References:
    Angelidaki, I. et al. (2009). Defining the biomethane potential (BMP).
    Water Science & Technology, 59(5), 927-934.
    Mata-Alvarez, J. et al. (2014). A critical review on anaerobic co-digestion
    achievements between 2010 and 2013. Renewable and Sustainable Energy Reviews, 36, 412-427.
    Buswell, A.M. & Mueller, H.F. (1952). Ind. Eng. Chem., 44(3), 550-552.
"""

import numpy as np


class AnaerobicDigesterF1b:
    """Mesophilic anaerobic digester with feedstock-specific BMP and co-digestion."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V = u["digester_volume_m3"]["value"]
        self.methane_base = u["methane_content_base"]["value"]
        self.k_hydrolysis = u["k_hydrolysis_per_day"]["value"]
        self.T_ref = u["T_ref_degC"]["value"] + 273.15  # K
        self.E_a = u["E_a_jmol"]["value"]
        self.R = u["R_jmolk"]["value"]
        self.LHV = u["LHV_methane_kwh_m3"]["value"]

        self.feedstock_db = params["feedstock_db"]
        self.cn_config = params["cn_ratio"]

    def _parse_feedstock(self, feedstock_type):
        """
        Parse feedstock input: either a string (single feedstock) or a dict
        of {feedstock_name: fraction} for co-digestion blends.

        Returns:
            List of (feedstock_name, fraction, feedstock_data) tuples
        """
        if isinstance(feedstock_type, str):
            if feedstock_type not in self.feedstock_db:
                raise ValueError(f"Unknown feedstock: {feedstock_type}. "
                                 f"Available: {list(self.feedstock_db.keys())}")
            return [(feedstock_type, 1.0, self.feedstock_db[feedstock_type])]

        elif isinstance(feedstock_type, dict):
            components = []
            total_frac = sum(feedstock_type.values())
            for name, frac in feedstock_type.items():
                if name not in self.feedstock_db:
                    raise ValueError(f"Unknown feedstock: {name}")
                norm_frac = frac / total_frac  # normalize to sum=1
                components.append((name, norm_frac, self.feedstock_db[name]))
            return components
        else:
            raise TypeError("feedstock_type must be str or dict")

    def blend_properties(self, feedstock_type):
        """
        Compute blended properties for a feedstock mix.

        Returns:
            dict with blended BMP, C/N ratio, methane content, k_hydrolysis, etc.
        """
        components = self._parse_feedstock(feedstock_type)

        BMP_blend = 0.0
        C_blend = 0.0
        N_blend = 0.0
        ch4_blend = 0.0
        k_blend = 0.0

        for name, frac, data in components:
            BMP_blend += frac * data["BMP_L_CH4_kgVS"]
            C_blend += frac * data["C_pct"]
            N_blend += frac * data["N_pct"]
            ch4_blend += frac * data["methane_content"]
            k_blend += frac * data["k_hydrolysis"]

        # C/N ratio
        cn_ratio = C_blend / N_blend if N_blend > 0 else 100.0

        return {
            "BMP": BMP_blend,
            "C_pct": C_blend,
            "N_pct": N_blend,
            "cn_ratio": cn_ratio,
            "methane_content": ch4_blend,
            "k_hydrolysis": k_blend,
            "n_components": len(components),
        }

    def co_digestion_synergy(self, cn_ratio, n_components):
        """
        Co-digestion synergy factor.
        - Single feedstock: factor = 1.0
        - Blends: up to 10% bonus when C/N is in optimal range [20, 30]
        """
        if n_components <= 1:
            return 1.0

        cn_opt = 25.0
        cn_min = self.cn_config["optimal_min"]
        cn_max = self.cn_config["optimal_max"]

        if cn_min <= cn_ratio <= cn_max:
            # In optimal range: synergy bonus proportional to proximity to 25
            proximity = 1.0 - abs(cn_ratio - cn_opt) / (cn_max - cn_min)
            return 1.0 + 0.10 * proximity  # up to 10% bonus
        else:
            return 1.0  # no synergy outside optimal range

    def cn_ratio_penalty(self, cn_ratio):
        """
        Yield penalty for C/N ratio outside optimal range.
        penalty = factor * |CN - 25| for CN outside [20, 30]
        Returns a multiplier in [0, 1].
        """
        cn_min = self.cn_config["optimal_min"]
        cn_max = self.cn_config["optimal_max"]
        penalty_factor = self.cn_config["penalty_factor"]

        if cn_min <= cn_ratio <= cn_max:
            return 1.0
        else:
            if cn_ratio < cn_min:
                deviation = cn_min - cn_ratio
            else:
                deviation = cn_ratio - cn_max
            return max(0.3, 1.0 - penalty_factor * deviation)

    def temperature_factor(self, temp_c):
        """Arrhenius temperature correction, normalized to T_ref."""
        T = float(temp_c) + 273.15
        f_T = np.exp(-self.E_a / self.R * (1.0 / T - 1.0 / self.T_ref))
        # Inhibition above 42 degC (transition to thermophilic)
        if temp_c > 42.0:
            f_inhibit = np.exp(-self.E_a / self.R * (1.0 / (42.0 + 273.15) - 1.0 / self.T_ref))
            f_T = f_inhibit * max(0.0, (55.0 - temp_c) / (55.0 - 42.0))
        return max(0.0, float(f_T))

    def vs_removal_efficiency(self, hrt_days, k_hydrolysis):
        """VS removal efficiency based on first-order kinetics."""
        hrt = float(hrt_days)
        k = float(k_hydrolysis)
        return 1.0 - np.exp(-k * hrt)

    def predict(self, feedstock_type, vs_loading_kg_m3_day,
                temperature_degC=37.0, hrt_days=20.0):
        """
        Compute biogas production for given feedstock and conditions.

        Args:
            feedstock_type:        str (single) or dict (blend with fractions)
            vs_loading_kg_m3_day:  VS loading rate [kgVS/(m3*day)]
            temperature_degC:      Digester temperature [degC]
            hrt_days:              Hydraulic retention time [days]

        Returns:
            dict with:
                biogas_yield_m3_day   : total biogas production [m3/day]
                methane_content_pct   : methane fraction [%]
                methane_yield_m3_day  : methane production [m3_CH4/day]
                vs_removal_pct        : VS removal efficiency [%]
                cn_ratio              : blend C/N ratio [-]
        """
        blend = self.blend_properties(feedstock_type)

        # Temperature correction
        f_T = self.temperature_factor(temperature_degC)

        # Kinetic yield factor (first-order, function of HRT and feedstock-specific k)
        k_eff = blend["k_hydrolysis"]
        yield_factor = 1.0 - np.exp(-k_eff * float(hrt_days))

        # C/N ratio effects
        cn_penalty = self.cn_ratio_penalty(blend["cn_ratio"])
        synergy = self.co_digestion_synergy(blend["cn_ratio"], blend["n_components"])

        # Effective BMP [L CH4/kgVS]
        BMP_eff = blend["BMP"] * yield_factor * f_T * cn_penalty * synergy

        # VS input rate [kgVS/day]
        VS_input = float(vs_loading_kg_m3_day) * self.V

        # Methane yield [m3 CH4/day] (BMP is in L, convert to m3)
        methane_yield = BMP_eff / 1000.0 * VS_input

        # Methane content [fraction]
        methane_content = blend["methane_content"]

        # Biogas yield [m3/day]
        biogas_yield = methane_yield / methane_content if methane_content > 0 else 0.0

        # VS removal [%]
        vs_removal = self.vs_removal_efficiency(hrt_days, k_eff) * 100.0

        return {
            "biogas_yield_m3_day": float(biogas_yield),
            "methane_content_pct": float(methane_content * 100.0),
            "methane_yield_m3_day": float(methane_yield),
            "vs_removal_pct": float(vs_removal),
            "cn_ratio": float(blend["cn_ratio"]),
        }
