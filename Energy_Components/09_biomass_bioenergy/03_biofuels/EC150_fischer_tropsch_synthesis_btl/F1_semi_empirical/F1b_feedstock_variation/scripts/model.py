"""
EC150 -- Fischer-Tropsch Synthesis (BtL) -- F1b Feedstock Variation Model

Builds on F1a (single syngas composition FT yield) by adding:
  - Feedstock-specific syngas quality from biomass gasification
  - Moisture-LHV coupling of the feed:
        LHV_eff = LHV_dry * (1 - M) - h_fg * M
  - H2/CO ratio effect on Anderson-Schulz-Flory (ASF) chain growth probability
  - Temperature effect on alpha (chain growth) and CO conversion
  - Part-load efficiency curve
  - Product selectivity: gasoline (C5-C12), diesel (C13-C20), wax (C21+)

FT Synthesis:
    (2n+1) H2 + n CO -> CnH(2n+2) + n H2O  (paraffin synthesis)
    Chain growth probability alpha from ASF distribution:
        W_n = n * (1-alpha)^2 * alpha^(n-1)
    Depends on H2/CO ratio and reactor temperature.

References:
    Anderson, R.B. (1984). The Fischer-Tropsch Synthesis. Academic Press.
    Davis, B.H. (2009). Fischer-Tropsch synthesis: relationship between iron catalyst
        composition and selectivity. Catalysis Today, 141, 25-33.
    van der Laan, G.P. & Beenackers, A.A.C.M. (1999). Kinetics and selectivity of the FTS.
        Catalysis Reviews, 41(3-4), 255-318.
    Dry, M.E. (2002). The Fischer-Tropsch process. Catalysis Today, 71, 227-241.
"""

import numpy as np

_H_FG = 2442.0   # kJ/kg latent heat

# FT syngas composition reference targets
_H2_CO_OPT = 2.0   # Optimal H2/CO molar ratio for FT (Dry 2002)


class FischerTropschF1b:
    """
    Fischer-Tropsch (BtL) with feedstock-specific syngas quality, moisture-LHV coupling,
    H2/CO-dependent ASF distribution, temperature effects, and part-load efficiency.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.T_ref        = u["T_ref_degC"]["value"]
        self.P_design     = u["P_design_MPa"]["value"]
        self.alpha_ref    = u["alpha_ref"]["value"]       # chain growth at design (H2/CO=2, T_ref)
        self.PLR_min      = u["PLR_min"]["value"]
        self.a0 = u["a0"]["value"]
        self.a1 = u["a1"]["value"]
        self.a2 = u["a2"]["value"]
        self.co_conversion_ref = u["CO_conversion_ref"]["value"]   # reference CO conversion
        self.feedstock_db = params["feedstock_db"]

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
        M  = np.clip(float(moisture_fraction), 0.0, 0.60)
        LHV_dry = fs["LHV_dry_MJ_kg"] * 1000.0
        LHV_eff = LHV_dry * (1.0 - M) - _H_FG * M
        return max(0.0, LHV_eff), LHV_dry

    def h2_co_ratio(self, feedstock_type, moisture_fraction):
        """
        Effective H2/CO ratio in syngas after gasification.
        Base ratio from feedstock, shifted by moisture (steam reforming increases H2).
        """
        fs  = self._get_feedstock(feedstock_type)
        M   = np.clip(float(moisture_fraction), 0.0, 0.60)
        h2co_base = fs.get("syngas_H2CO_ratio", 1.8)
        # Moisture promotes water-gas shift: each 10% moisture adds ~0.15 to H2/CO
        h2co = h2co_base + 1.5 * M
        return float(np.clip(h2co, 0.5, 4.0))

    def alpha(self, temperature_degC, h2_co):
        """
        ASF chain growth probability alpha.
        - Decreases with T (higher T -> lighter products, lower alpha)
        - Increases with H2/CO (more H2 promotes chain growth for Co catalyst)
        - Range: ~0.65 (gasoline) to ~0.92 (wax, low T LTFT)
        (van der Laan 1999)
        """
        T     = float(temperature_degC)
        ratio = float(h2_co)

        # Temperature effect: alpha decreases linearly with T above T_ref
        dT = T - self.T_ref
        alpha_T = self.alpha_ref - 0.002 * dT  # 0.002 per degC (Dry 2002)

        # H2/CO ratio effect: deviation from optimal H2/CO=2
        dR = ratio - _H2_CO_OPT
        alpha_R = alpha_T - 0.03 * abs(dR)    # penalty for off-ratio

        return float(np.clip(alpha_R, 0.40, 0.95))

    def product_selectivity(self, alpha):
        """
        ASF product selectivity using chain growth probability alpha.
        Groups: CH4 (C1), LPG (C2-C4), gasoline (C5-C12), diesel (C13-C20), wax (C21+).

        W_n = n * (1-alpha)^2 * alpha^(n-1)   [Anderson 1984]
        Mass fractions summed over carbon number ranges.
        """
        a = float(np.clip(alpha, 0.40, 0.95))
        ia = 1.0 - a

        def wn(n):
            return n * (ia ** 2) * (a ** (n - 1))

        ch4    = float(wn(1))
        lpg    = float(sum(wn(n) for n in range(2, 5)))    # C2-C4
        gasoline = float(sum(wn(n) for n in range(5, 13)))  # C5-C12
        diesel = float(sum(wn(n) for n in range(13, 21)))   # C13-C20
        wax    = float(sum(wn(n) for n in range(21, 51)))   # C21-C50 (truncated)

        total = ch4 + lpg + gasoline + diesel + wax
        if total > 0:
            ch4      /= total
            lpg      /= total
            gasoline /= total
            diesel   /= total
            wax      /= total

        return {"CH4": ch4, "LPG": lpg, "gasoline": gasoline, "diesel": diesel, "wax": wax}

    def co_conversion(self, temperature_degC, h2_co):
        """
        CO conversion efficiency at given conditions.
        Decreases away from optimal T and H2/CO=2.
        """
        T     = float(temperature_degC)
        ratio = float(h2_co)

        # Temperature effect: bell-shaped around T_ref
        dT = abs(T - self.T_ref)
        f_T = np.exp(-0.0002 * dT ** 2)

        # H2/CO ratio effect
        dR = ratio - _H2_CO_OPT
        f_R = np.exp(-0.05 * dR ** 2)

        return float(np.clip(self.co_conversion_ref * f_T * f_R, 0.20, 0.95))

    def ft_liquid_yield(self, feedstock_type, temperature_degC, moisture_fraction):
        """
        FT liquid yield [kg liquids / kg dry biomass].
        Combines gasification yield, syngas LHV, CO conversion, and product selectivity.
        """
        fs  = self._get_feedstock(feedstock_type)
        h2co = self.h2_co_ratio(feedstock_type, moisture_fraction)
        a    = self.alpha(temperature_degC, h2co)
        conv = self.co_conversion(temperature_degC, h2co)
        sel  = self.product_selectivity(a)

        # Syngas-to-liquid efficiency: fraction of syngas energy converted to liquids
        # Typical: 40-55% cold gas to liquid (Dry 2002, Davis 2009)
        liquid_fraction = sel["gasoline"] + sel["diesel"] + sel["wax"]

        # Base yield from feedstock carbon content and syngas yield
        C_frac = fs.get("C_frac", 0.50)     # carbon fraction
        syngas_yield = fs.get("syngas_yield_nm3_kg_dry", 2.5)  # Nm3/kg dry

        # Simplified: FT liquid yield = C_frac * syngas_efficiency * conversion * liquid_frac
        Y_liquid = C_frac * 0.45 * conv * liquid_fraction
        return float(np.clip(Y_liquid, 0.0, 0.40))

    def thermal_efficiency(self, PLR, moisture_fraction, feedstock_type):
        """Process thermal efficiency = (a0 + a1*PLR + a2*PLR^2) * moisture_factor."""
        PLR_eff  = np.clip(np.asarray(PLR, dtype=float), self.PLR_min, 1.0)
        eta_base = self.a0 + self.a1 * PLR_eff + self.a2 * PLR_eff ** 2
        LHV_eff, LHV_dry = self.lhv_effective(feedstock_type, moisture_fraction)
        mf = LHV_eff / LHV_dry if LHV_dry > 0 else 0.0
        return float(np.clip(eta_base * mf, 0.0, 1.0))

    def predict(self, feedstock_type, temperature_degC=230.0, moisture_fraction=0.10,
                PLR=1.0, feed_rate_kg_h=1000.0):
        """
        Compute FT synthesis outputs.

        Args:
            feedstock_type:    str from feedstock_db
            temperature_degC:  FT reactor temperature [degC] (180-350)
            moisture_fraction: Feed wet-basis moisture [0-0.60]
            PLR:               Part-load ratio [PLR_min-1.0]
            feed_rate_kg_h:    Dry feed rate [kg/h]

        Returns:
            dict with:
                ft_liquid_yield       : kg liquids/kg_dry_feed [-]
                product_selectivity   : dict {CH4, LPG, gasoline, diesel, wax}
                alpha                 : ASF chain growth probability [-]
                co_conversion         : CO conversion efficiency [-]
                h2_co_ratio           : effective H2/CO in syngas [-]
                LHV_eff_MJ_kg         : effective feed LHV [MJ/kg wet]
                moisture_lhv_factor   : LHV_eff/LHV_dry [-]
                thermal_efficiency    : process efficiency [-]
                ft_liquid_rate_kg_h   : total liquid production [kg/h]
                diesel_rate_kg_h      : diesel fraction [kg/h]
        """
        PLR_eff = float(np.clip(PLR, self.PLR_min, 1.0))
        h2co    = self.h2_co_ratio(feedstock_type, moisture_fraction)
        a       = self.alpha(temperature_degC, h2co)
        conv    = self.co_conversion(temperature_degC, h2co)
        sel     = self.product_selectivity(a)
        Y_liq   = self.ft_liquid_yield(feedstock_type, temperature_degC, moisture_fraction)
        LHV_eff, LHV_dry = self.lhv_effective(feedstock_type, moisture_fraction)
        mf      = LHV_eff / LHV_dry if LHV_dry > 0 else 0.0
        eta_th  = float(self.thermal_efficiency(PLR_eff, moisture_fraction, feedstock_type))

        m_feed_eff = float(feed_rate_kg_h) * PLR_eff

        return {
            "ft_liquid_yield":      Y_liq,
            "product_selectivity":  {k: float(v) for k, v in sel.items()},
            "alpha":                a,
            "co_conversion":        conv,
            "h2_co_ratio":          h2co,
            "LHV_eff_MJ_kg":        LHV_eff / 1000.0,
            "moisture_lhv_factor":  mf,
            "thermal_efficiency":   eta_th,
            "ft_liquid_rate_kg_h":  m_feed_eff * Y_liq,
            "diesel_rate_kg_h":     m_feed_eff * Y_liq * sel["diesel"],
        }
