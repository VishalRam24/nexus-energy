"""
EC147 -- Hydrothermal Liquefaction (HTL) -- F1b Feedstock Variation Model

Builds on F1a (single-point bio-crude yield) by adding:
  - Feedstock-specific biochemical composition (lipid/protein/carbohydrate/ash)
  - Moisture-LHV coupling (HTL advantageously uses wet feedstocks; LHV coupling
    captures the energy content of the dried equivalent)
        LHV_eff = LHV_dry * (1 - M) - h_fg * M
  - Temperature effect on bio-crude yield and energy recovery
  - Part-load thermal efficiency curve
  - Product phase distribution: bio-crude, aqueous, gas, solid residue

HTL uniquely advantages wet feedstocks (algae, sludge) because no pre-drying is
needed — supercritical/subcritical water acts as reaction medium.

Bio-crude yield model (empirical, Peterson et al. 2008; Anastasakis & Ross 2011):
    Y_bc = (0.35*lipid + 0.18*protein + 0.12*carb) * f_T * f_moisture_benefit
    Energy recovery (ER) = Y_bc * HHV_biocrude / HHV_dry

f_T: Arrhenius-type temperature factor peaking ~330-350 degC
f_moisture_benefit: wet feedstocks benefit from subcritical water reaction medium

References:
    Peterson, A.A. et al. (2008). Thermochemical biofuel production in hydrothermal media.
        Energy & Environmental Science, 1, 32-65.
    Anastasakis, K. & Ross, A.B. (2011). Hydrothermal liquefaction of the brown macro-alga
        Laminaria saccharina. Bioresource Technology, 102, 4876-4883.
    Vardon, D.R. et al. (2011). Chemical properties of biocrude oil from subcritical water HTL.
        J. Analytical and Applied Pyrolysis, 91, 108-117.
    Elliott, D.C. et al. (2015). Hydrothermal liquefaction of biomass. Biofuels, Bioproducts
        and Biorefining, 9, 507-527.
"""

import numpy as np

_H_FG = 2442.0   # kJ/kg latent heat at 25 degC
_E_A  = 55000.0  # J/mol activation energy for HTL (Peterson et al. 2008)
_R    = 8.314    # J/(mol*K)
_T_OPT = 330.0   # degC optimal HTL temperature


class HTLF1b:
    """
    HTL reactor with feedstock-specific biochemical composition, moisture-LHV
    coupling, temperature-dependent bio-crude yield, and part-load efficiency.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.T_ref        = u["T_ref_degC"]["value"]        # degC reference T
        self.P_design     = u["P_design_MPa"]["value"]      # MPa
        self.PLR_min      = u["PLR_min"]["value"]
        self.a0 = u["a0"]["value"]
        self.a1 = u["a1"]["value"]
        self.a2 = u["a2"]["value"]
        self.HHV_biocrude = u["HHV_biocrude_MJ_kg"]["value"]  # MJ/kg (typical 33-37 MJ/kg)
        self.feedstock_db = params["feedstock_db"]

    def _get_feedstock(self, name):
        if name not in self.feedstock_db:
            raise ValueError(
                f"Unknown feedstock: {name}. "
                f"Available: {list(self.feedstock_db.keys())}"
            )
        return self.feedstock_db[name]

    def lhv_effective(self, feedstock_type, moisture_fraction):
        """
        LHV_eff [kJ/kg wet] = LHV_dry*(1-M) - h_fg*M
        Note: HTL can process wet feedstocks (M up to 80%).
        """
        fs = self._get_feedstock(feedstock_type)
        M  = np.clip(float(moisture_fraction), 0.0, 0.90)
        LHV_dry = fs["LHV_dry_MJ_kg"] * 1000.0
        LHV_eff = LHV_dry * (1.0 - M) - _H_FG * M
        return max(0.0, LHV_eff), LHV_dry

    def temperature_factor(self, temperature_degC):
        """
        Arrhenius-based temperature factor normalized to optimal T.
        Peaks at ~330-350 degC; above T_opt secondary decomposition reduces yield.
        f_T = exp(-E_a/R * (1/T - 1/T_opt)) for T < T_opt
            = exp(-k_decomp * (T - T_opt)) for T > T_opt
        """
        T    = float(temperature_degC)
        T_K  = T + 273.15
        T_opt_K = _T_OPT + 273.15

        if T <= _T_OPT:
            f = np.exp(-_E_A / _R * (1.0 / T_K - 1.0 / T_opt_K))
        else:
            # Decomposition above optimal (secondary cracking -> gas)
            k_decomp = 0.015   # 1/degC  (Elliott et al. 2015)
            f = np.exp(-k_decomp * (T - _T_OPT))

        return float(np.clip(f, 0.05, 1.0))

    def biocrude_yield(self, feedstock_type, temperature_degC, moisture_fraction):
        """
        Bio-crude mass yield [kg_bc/kg_dry_feed].

        Composition-based: lipids convert most efficiently, then proteins,
        then carbohydrates (Vardon et al. 2011).

        Moisture benefit: subcritical water improves extraction for wet feedstocks.
        Peak benefit at M ~ 80% (algae).
        """
        fs  = self._get_feedstock(feedstock_type)
        T   = np.clip(float(temperature_degC), 250.0, 400.0)
        M   = np.clip(float(moisture_fraction), 0.0, 0.90)

        lipid   = fs.get("lipid_frac", 0.10)
        protein = fs.get("protein_frac", 0.20)
        carb    = fs.get("carb_frac", 0.40)

        # Base yield from biochemical composition
        Y_base = 0.45 * lipid + 0.25 * protein + 0.12 * carb

        # Temperature factor
        f_T = self.temperature_factor(T)

        # Moisture benefit for wet feedstocks (subcritical water medium)
        # Slight benefit peaks at M~0.8 then negligible past that
        f_moisture = 1.0 + 0.08 * min(M / 0.80, 1.0)

        Y_bc = Y_base * f_T * f_moisture
        return float(np.clip(Y_bc, 0.01, 0.55))

    def product_distribution(self, feedstock_type, temperature_degC, moisture_fraction):
        """
        Product phase distribution [kg/kg_dry_feed]:
          - bio_crude
          - aqueous (dissolved organics + nutrients)
          - gas (CO2, CH4, H2)
          - solid_residue (char, mineral ash)

        Aqueous increases with protein (N-compounds dissolve).
        Gas increases above 350 degC.
        Solid_residue ~ ash content.
        """
        fs  = self._get_feedstock(feedstock_type)
        T   = np.clip(float(temperature_degC), 250.0, 400.0)
        M   = np.clip(float(moisture_fraction), 0.0, 0.90)

        Y_bc  = self.biocrude_yield(feedstock_type, T, M)
        ash   = fs.get("ash_frac", 0.05)
        protein = fs.get("protein_frac", 0.20)

        # Solid residue ~ ash (mineral matter retained)
        solid = float(np.clip(ash * 0.85, 0.01, 0.25))

        # Gas increases with T above 300 degC
        gas_factor = max(0.0, (T - 300.0) / 100.0) * 0.10
        gas = float(np.clip(0.05 + gas_factor, 0.03, 0.25))

        # Aqueous ~ dissolved protein metabolites and organic acids
        aqueous = float(np.clip(0.25 * protein + 0.10, 0.10, 0.40))

        # Ensure normalization
        total = Y_bc + solid + gas + aqueous
        if total > 1.0:
            # Scale down proportionally
            Y_bc    /= total
            solid   /= total
            gas     /= total
            aqueous /= total

        return {
            "bio_crude": Y_bc,
            "aqueous":   aqueous,
            "gas":       gas,
            "solid":     solid,
        }

    def energy_recovery(self, feedstock_type, temperature_degC, moisture_fraction):
        """
        Energy recovery = Y_bc * HHV_biocrude / HHV_dry_feed.
        """
        fs = self._get_feedstock(feedstock_type)
        Y_bc = self.biocrude_yield(feedstock_type, temperature_degC, moisture_fraction)
        _, LHV_dry = self.lhv_effective(feedstock_type, moisture_fraction)
        ER = Y_bc * self.HHV_biocrude * 1000.0 / LHV_dry if LHV_dry > 0 else 0.0
        return float(np.clip(ER, 0.0, 1.0))

    def thermal_efficiency(self, PLR, moisture_fraction, feedstock_type):
        """Reactor thermal efficiency with part-load polynomial and moisture factor."""
        PLR_eff = np.clip(np.asarray(PLR, dtype=float), self.PLR_min, 1.0)
        eta_base = self.a0 + self.a1 * PLR_eff + self.a2 * PLR_eff ** 2
        LHV_eff, LHV_dry = self.lhv_effective(feedstock_type, moisture_fraction)
        mf = LHV_eff / LHV_dry if LHV_dry > 0 else 0.0
        # For very wet feedstocks, use energy density of wet feed vs dry
        return float(np.clip(eta_base * max(mf, 0.1), 0.0, 1.0))

    def predict(self, feedstock_type, temperature_degC, moisture_fraction=0.70,
                PLR=1.0, feed_rate_kg_h=500.0):
        """
        Compute HTL outputs.

        Args:
            feedstock_type:    str from feedstock_db
            temperature_degC:  Reactor temperature [degC] (250-400)
            moisture_fraction: Wet-basis moisture (0-0.90); HTL typical 0.70-0.90
            PLR:               Part-load ratio [PLR_min - 1.0]
            feed_rate_kg_h:    Wet feed rate [kg/h]

        Returns:
            dict with:
                bio_crude_yield    : kg_bc/kg_dry_feed [-]
                product_distribution: dict {bio_crude, aqueous, gas, solid} kg/kg_dry
                energy_recovery    : fraction of dry feed energy in bio-crude [-]
                LHV_eff_MJ_kg      : effective feed LHV [MJ/kg wet]
                moisture_lhv_factor: LHV_eff/LHV_dry [-]
                thermal_efficiency : reactor efficiency [-]
                bio_crude_rate_kg_h: bio-crude production [kg/h based on dry feed]
        """
        PLR_eff = float(np.clip(PLR, self.PLR_min, 1.0))
        Y_bc  = self.biocrude_yield(feedstock_type, temperature_degC, moisture_fraction)
        prods = self.product_distribution(feedstock_type, temperature_degC, moisture_fraction)
        ER    = self.energy_recovery(feedstock_type, temperature_degC, moisture_fraction)
        LHV_eff, LHV_dry = self.lhv_effective(feedstock_type, moisture_fraction)
        mf    = LHV_eff / LHV_dry if LHV_dry > 0 else 0.0
        eta_th = float(self.thermal_efficiency(PLR_eff, moisture_fraction, feedstock_type))

        # Effective dry feed rate (wet feed * (1-M) * PLR)
        M = np.clip(float(moisture_fraction), 0.0, 0.90)
        dry_rate = float(feed_rate_kg_h) * (1.0 - M) * PLR_eff

        return {
            "bio_crude_yield":      Y_bc,
            "product_distribution": {k: float(v) for k, v in prods.items()},
            "energy_recovery":      ER,
            "LHV_eff_MJ_kg":        LHV_eff / 1000.0,
            "moisture_lhv_factor":  mf,
            "thermal_efficiency":   eta_th,
            "bio_crude_rate_kg_h":  dry_rate * Y_bc,
        }
