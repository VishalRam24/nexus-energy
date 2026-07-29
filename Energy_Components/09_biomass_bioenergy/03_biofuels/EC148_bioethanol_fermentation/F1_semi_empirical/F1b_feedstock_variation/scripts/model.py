"""
EC148 -- Bioethanol Fermentation -- F1b Feedstock Variation Model

Builds on F1a (single-feedstock ethanol yield) by adding:
  - Feedstock-specific fermentable sugar content (cellulose, starch, sucrose)
  - Moisture-LHV coupling for the feed:
        LHV_eff = LHV_dry * (1 - M) - h_fg * M
  - Temperature effect on fermentation yield (Arrhenius kinetics, optimal 30-35 degC)
  - Pretreatment efficiency for lignocellulosic feedstocks
  - Part-load efficiency curve for the overall process
  - Inhibitor penalty for high solids loading (product inhibition)

Biochemical stoichiometry (Gay-Lussac):
    C6H12O6 -> 2 C2H5OH + 2 CO2
    Theoretical yield: 0.511 kg EtOH / kg glucose

For lignocellulosic feedstocks, pretreatment converts cellulose to glucose
with a pretreatment efficiency factor eta_pretreat (0.5-0.85).

References:
    Balat, M. et al. (2008). Progress in bioethanol processing.
        Progress in Energy and Combustion Science, 34, 551-573.
    Chandel, A.K. et al. (2011). The path forward for lignocellulose biorefineries.
        Bioresource Technology, 102, 11-21.
    Kumar, D. & Murthy, G.S. (2011). Life cycle assessment of energy and GHG emissions
        during ethanol production. GCB Bioenergy, 3, 517-533.
    Wyman, C.E. (1999). Biomass ethanol: technical progress, opportunities, and commercial
        challenges. Annual Review of Energy and the Environment, 24, 189-226.
"""

import numpy as np

_H_FG = 2442.0       # kJ/kg latent heat
_THEO_YIELD = 0.511  # kg EtOH / kg fermentable glucose (Gay-Lussac stoichiometry)
_E_A = 40000.0       # J/mol activation energy for yeast fermentation
_R   = 8.314         # J/(mol*K)
_T_OPT = 32.0        # degC optimal fermentation temperature


class BioethanolFermentationF1b:
    """
    Bioethanol fermentation with feedstock-specific sugar content, moisture-LHV
    coupling, temperature kinetics, and part-load efficiency.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.PLR_min      = u["PLR_min"]["value"]
        self.a0 = u["a0"]["value"]
        self.a1 = u["a1"]["value"]
        self.a2 = u["a2"]["value"]
        self.yeast_efficiency = u["yeast_efficiency"]["value"]  # fraction of theoretical yield
        self.k_inhibition     = u["k_inhibition"]["value"]      # product inhibition coefficient
        self.C_inhibit_50     = u["C_inhibit_50_pct"]["value"]  # EtOH concentration at 50% inhibition
        self.feedstock_db     = params["feedstock_db"]

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
        M  = np.clip(float(moisture_fraction), 0.0, 0.85)
        LHV_dry = fs["LHV_dry_MJ_kg"] * 1000.0
        LHV_eff = LHV_dry * (1.0 - M) - _H_FG * M
        return max(0.0, LHV_eff), LHV_dry

    def temperature_factor(self, temperature_degC):
        """
        Yeast fermentation yield factor vs temperature.
        Arrhenius activation below T_opt; thermal inactivation above T_opt.
        Optimal: ~30-35 degC (Saccharomyces cerevisiae).
        Above 40 degC: significant cell death.
        """
        T = float(temperature_degC)
        T_K    = T + 273.15
        T_opt_K = _T_OPT + 273.15

        if T <= _T_OPT:
            # Below optimal: Arrhenius activation
            f = np.exp(-_E_A / _R * (1.0 / T_K - 1.0 / T_opt_K))
            f = max(0.05, float(f))
        else:
            # Above optimal: sigmoidal inactivation
            k_inact = 0.10   # 1/degC
            f = np.exp(-k_inact * (T - _T_OPT))
            f = max(0.0, float(f))

        return float(np.clip(f, 0.0, 1.0))

    def pretreatment_efficiency(self, feedstock_type):
        """
        Fraction of cellulose/starch converted to fermentable sugars after pretreatment.
        Lignocellulosic feedstocks require more severe pretreatment (lower efficiency).
        """
        fs = self._get_feedstock(feedstock_type)
        return float(fs.get("pretreatment_efficiency", 0.85))

    def fermentable_sugar_fraction(self, feedstock_type):
        """
        Effective fermentable sugar fraction [kg glucose_equiv/kg dry biomass].
        Includes direct sugars (sucrose, starch) and cellulose * pretreatment_efficiency.
        """
        fs = self._get_feedstock(feedstock_type)
        eta_pt = self.pretreatment_efficiency(feedstock_type)

        # Direct fermentable (starch, sucrose hydrolysis ~100% efficient)
        direct = fs.get("starch_frac", 0.0) + fs.get("sucrose_frac", 0.0)

        # Cellulose via pretreatment (lignocellulosic route)
        cellulose_ferm = fs.get("cellulose_frac", 0.0) * eta_pt

        return float(np.clip(direct + cellulose_ferm, 0.0, 1.0))

    def inhibition_factor(self, ethanol_conc_pct):
        """
        Product inhibition factor (Andrews-type).
        f_inhib = C_50 / (C_50 + C_EtOH)
        """
        C = max(0.0, float(ethanol_conc_pct))
        return float(self.C_inhibit_50 / (self.C_inhibit_50 + C))

    def ethanol_yield(self, feedstock_type, temperature_degC, moisture_fraction,
                      ethanol_conc_pct=8.0):
        """
        Ethanol mass yield [kg EtOH / kg dry biomass].
        Y = sugar_frac * theo_yield * yeast_eff * f_T * f_inhibit
        """
        sugar = self.fermentable_sugar_fraction(feedstock_type)
        f_T   = self.temperature_factor(temperature_degC)
        f_inh = self.inhibition_factor(ethanol_conc_pct)

        Y = sugar * _THEO_YIELD * self.yeast_efficiency * f_T * f_inh
        return float(np.clip(Y, 0.0, 0.50))

    def thermal_efficiency(self, PLR, moisture_fraction, feedstock_type):
        """Overall process thermal efficiency = (a0 + a1*PLR + a2*PLR^2) * moisture_factor."""
        PLR_eff  = np.clip(np.asarray(PLR, dtype=float), self.PLR_min, 1.0)
        eta_base = self.a0 + self.a1 * PLR_eff + self.a2 * PLR_eff ** 2
        LHV_eff, LHV_dry = self.lhv_effective(feedstock_type, moisture_fraction)
        mf = LHV_eff / LHV_dry if LHV_dry > 0 else 0.0
        return float(np.clip(eta_base * mf, 0.0, 1.0))

    def predict(self, feedstock_type, temperature_degC=32.0, moisture_fraction=0.20,
                PLR=1.0, feed_rate_kg_h=1000.0, ethanol_conc_pct=8.0):
        """
        Compute bioethanol fermentation outputs.

        Args:
            feedstock_type:    str from feedstock_db
            temperature_degC:  Fermentation temperature [degC] (20-45)
            moisture_fraction: Feed wet-basis moisture [0-0.85]
            PLR:               Part-load ratio [PLR_min-1.0]
            feed_rate_kg_h:    Dry feed rate [kg/h]
            ethanol_conc_pct:  Product ethanol concentration [%v/v] (default 8%)

        Returns:
            dict with:
                ethanol_yield      : kg EtOH/kg dry feed [-]
                sugar_fraction     : fermentable sugar fraction [-]
                temperature_factor : yeast kinetic factor [-]
                pretreatment_eff   : pretreatment efficiency [-]
                LHV_eff_MJ_kg      : effective feed LHV [MJ/kg wet]
                moisture_lhv_factor: LHV_eff/LHV_dry [-]
                thermal_efficiency : process efficiency [-]
                ethanol_rate_kg_h  : ethanol production [kg/h]
        """
        PLR_eff = float(np.clip(PLR, self.PLR_min, 1.0))
        Y_eth   = self.ethanol_yield(feedstock_type, temperature_degC,
                                     moisture_fraction, ethanol_conc_pct)
        sugar   = self.fermentable_sugar_fraction(feedstock_type)
        f_T     = self.temperature_factor(temperature_degC)
        eta_pt  = self.pretreatment_efficiency(feedstock_type)
        LHV_eff, LHV_dry = self.lhv_effective(feedstock_type, moisture_fraction)
        mf      = LHV_eff / LHV_dry if LHV_dry > 0 else 0.0
        eta_th  = float(self.thermal_efficiency(PLR_eff, moisture_fraction, feedstock_type))

        eth_rate = float(feed_rate_kg_h) * PLR_eff * Y_eth

        return {
            "ethanol_yield":       Y_eth,
            "sugar_fraction":      sugar,
            "temperature_factor":  f_T,
            "pretreatment_eff":    eta_pt,
            "LHV_eff_MJ_kg":       LHV_eff / 1000.0,
            "moisture_lhv_factor": mf,
            "thermal_efficiency":  eta_th,
            "ethanol_rate_kg_h":   eth_rate,
        }
