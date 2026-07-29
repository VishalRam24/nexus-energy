"""
EC146 -- Torrefaction Reactor -- F1b Feedstock Variation Model

Builds on F1a (single-point solid yield/LHV) by adding:
  - Feedstock-specific initial LHV and composition (cellulose/hemicellulose/lignin)
  - Moisture-LHV coupling pre-torrefaction:
        LHV_eff_feed = LHV_dry * (1 - M) - h_fg * M
  - Temperature-dependent mass yield and energy densification
  - Residence time effect on torrefaction severity
  - Part-load thermal efficiency curve

Torrefaction (mild pyrolysis at 200-300 degC) degrades hemicellulose, making the
solid product (torrefied biomass, "biocoal") more brittle and energy-dense.

Energy densification ratio (EDR) and mass yield (MY):
    MY  = exp(-k_m * (T - T_ref)^n_m * t^p_m)  -- empirical, Bach et al. 2017
    EDR = HHV_torrefied / HHV_raw
        = 1 + a_edr * (T - 200) * (t/30)^0.3  -- linear with temp above 200 degC

Hemicellulose fraction dominates mass loss:
    f_hc_loss = min(1, (T - 200) / 100)  [fraction of hemicellulose degraded]

References:
    Bach, Q.V. et al. (2017). Torrefaction kinetics. Fuel, 202, 573-578.
    Tran, K.Q. et al. (2016). A kinetic study on wet torrefaction. Applied Energy, 162, 1392-1399.
    van der Stelt, M.J.C. et al. (2011). Biomass upgrading by torrefaction.
        Biomass & Bioenergy, 35(9), 3748-3762.
    Bergman, P.C.A. et al. (2005). Torrefaction for biomass co-firing in coal-fired
        power plants. ECN-C-05-013.
"""

import numpy as np

_H_FG = 2442.0   # kJ/kg latent heat at 25 degC


class TorrefactionReactorF1b:
    """
    Torrefaction reactor with feedstock-specific composition, moisture-LHV coupling,
    temperature/residence-time effects, and part-load efficiency.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.T_ref       = u["T_ref_degC"]["value"]      # reference T (degC)
        self.t_ref       = u["t_ref_min"]["value"]       # reference residence time (min)
        self.k_m         = u["k_m"]["value"]             # mass yield rate coeff
        self.n_m         = u["n_m"]["value"]             # temperature exponent
        self.p_m         = u["p_m"]["value"]             # residence time exponent
        self.a_edr       = u["a_edr"]["value"]           # EDR slope coefficient
        self.PLR_min     = u["PLR_min"]["value"]
        self.a0 = u["a0"]["value"]
        self.a1 = u["a1"]["value"]
        self.a2 = u["a2"]["value"]
        self.Q_rated     = u["Q_rated_kw"]["value"]
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
        LHV_eff [kJ/kg wet] = LHV_dry * (1-M) - h_fg * M
        Feed moisture must be < 15% for effective torrefaction (van der Stelt 2011).
        """
        fs = self._get_feedstock(feedstock_type)
        M = np.clip(float(moisture_fraction), 0.0, 0.30)
        LHV_dry = fs["LHV_dry_MJ_kg"] * 1000.0
        LHV_eff = LHV_dry * (1.0 - M) - _H_FG * M
        return max(0.0, LHV_eff), LHV_dry

    def mass_yield(self, feedstock_type, temperature_degC, residence_time_min):
        """
        Solid mass yield [kg_torrefied / kg_dry_feed].
        Decreases with temperature and residence time.
        MY = exp(-k_m * dT^n_m * t^p_m)
        where dT = T - T_ref (minimum 0).
        Hemicellulose-rich feedstocks degrade faster.
        """
        fs = self._get_feedstock(feedstock_type)
        T  = np.clip(float(temperature_degC), 200.0, 300.0)
        t  = np.clip(float(residence_time_min), 5.0, 120.0)
        dT = max(0.0, T - self.T_ref)

        # Hemicellulose fraction increases mass loss rate
        hc  = fs.get("hemicellulose_frac", 0.30)
        k_m_eff = self.k_m * (1.0 + 0.5 * (hc - 0.30))

        MY = np.exp(-k_m_eff * (dT ** self.n_m) * (t ** self.p_m))
        return float(np.clip(MY, 0.30, 1.0))

    def energy_densification_ratio(self, feedstock_type, temperature_degC, residence_time_min):
        """
        EDR = HHV_torrefied / HHV_raw.
        Increases with T and residence time; higher lignin -> slightly higher EDR.
        EDR = 1 + a_edr * (T - 200) * (t/30)^0.3 * lignin_factor
        """
        fs  = self._get_feedstock(feedstock_type)
        T   = np.clip(float(temperature_degC), 200.0, 300.0)
        t   = np.clip(float(residence_time_min), 5.0, 120.0)
        lign = fs.get("lignin_frac", 0.25)

        lignin_factor = 1.0 + 0.3 * (lign - 0.25)
        EDR = 1.0 + self.a_edr * (T - 200.0) * (t / 30.0) ** 0.3 * lignin_factor
        return float(np.clip(EDR, 1.0, 1.35))   # max ~35% energy densification

    def energy_yield(self, feedstock_type, temperature_degC, residence_time_min):
        """
        Energy yield [%] = MY * EDR * 100.
        Fraction of feed energy retained in torrefied solid.
        Typically 85-95% (Bergman 2005).
        """
        MY  = self.mass_yield(feedstock_type, temperature_degC, residence_time_min)
        EDR = self.energy_densification_ratio(feedstock_type, temperature_degC, residence_time_min)
        return float(np.clip(MY * EDR, 0.0, 1.05))

    def torrefied_lhv(self, feedstock_type, temperature_degC, residence_time_min):
        """LHV of torrefied product [MJ/kg_torrefied]."""
        fs  = self._get_feedstock(feedstock_type)
        EDR = self.energy_densification_ratio(feedstock_type, temperature_degC, residence_time_min)
        return float(fs["LHV_dry_MJ_kg"] * EDR)

    def thermal_efficiency(self, PLR, moisture_fraction, feedstock_type):
        """
        Reactor thermal efficiency = (a0 + a1*PLR + a2*PLR^2) * moisture_factor.
        """
        PLR_eff = np.clip(np.asarray(PLR, dtype=float), self.PLR_min, 1.0)
        eta_base = self.a0 + self.a1 * PLR_eff + self.a2 * PLR_eff ** 2
        LHV_eff, LHV_dry = self.lhv_effective(feedstock_type, moisture_fraction)
        mf = LHV_eff / LHV_dry if LHV_dry > 0 else 0.0
        return float(np.clip(eta_base * mf, 0.0, 1.0))

    def predict(self, feedstock_type, temperature_degC, residence_time_min=30.0,
                moisture_fraction=0.10, PLR=1.0, feed_rate_kg_h=1000.0):
        """
        Compute torrefaction outputs.

        Args:
            feedstock_type:      str from feedstock_db
            temperature_degC:    Reactor temperature [degC] (200-300)
            residence_time_min:  Residence time [min] (5-120)
            moisture_fraction:   Feed wet-basis moisture [0-0.30]
            PLR:                 Part-load ratio [PLR_min - 1.0]
            feed_rate_kg_h:      Dry feed rate [kg/h]

        Returns:
            dict with:
                mass_yield          : kg_solid/kg_dry_feed [-]
                energy_densification: EDR [-]
                energy_yield        : fraction of feed energy retained [-]
                torrefied_LHV_MJ_kg : LHV of product [MJ/kg]
                LHV_eff_MJ_kg       : effective feed LHV [MJ/kg wet]
                moisture_lhv_factor : LHV_eff/LHV_dry [-]
                thermal_efficiency  : reactor thermal efficiency [-]
                solid_rate_kg_h     : torrefied solid production rate [kg/h]
        """
        PLR_eff = float(np.clip(PLR, self.PLR_min, 1.0))
        MY  = self.mass_yield(feedstock_type, temperature_degC, residence_time_min)
        EDR = self.energy_densification_ratio(feedstock_type, temperature_degC, residence_time_min)
        EY  = self.energy_yield(feedstock_type, temperature_degC, residence_time_min)
        LHV_torr = self.torrefied_lhv(feedstock_type, temperature_degC, residence_time_min)
        LHV_eff, LHV_dry = self.lhv_effective(feedstock_type, moisture_fraction)
        mf  = LHV_eff / LHV_dry if LHV_dry > 0 else 0.0
        eta_th = float(self.thermal_efficiency(PLR_eff, moisture_fraction, feedstock_type))

        solid_rate = float(feed_rate_kg_h) * PLR_eff * MY

        return {
            "mass_yield":          MY,
            "energy_densification": EDR,
            "energy_yield":        EY,
            "torrefied_LHV_MJ_kg": LHV_torr,
            "LHV_eff_MJ_kg":       LHV_eff / 1000.0,
            "moisture_lhv_factor": mf,
            "thermal_efficiency":  eta_th,
            "solid_rate_kg_h":     solid_rate,
        }
