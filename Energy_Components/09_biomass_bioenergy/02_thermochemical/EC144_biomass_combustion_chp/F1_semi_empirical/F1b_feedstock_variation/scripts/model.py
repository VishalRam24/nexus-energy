"""
EC144 -- Biomass Combustion CHP -- F1b Feedstock Variation Model

Extends F1a (fixed efficiency) with:
  - Feedstock-specific LHV and ash content
  - Moisture-LHV coupling (EC087 convention):
        LHV_eff = LHV_dry * (1 - M) - h_fg * M
  - Part-load efficiency curve: eta(PLR) = (a0 + a1*PLR + a2*PLR^2) * moisture_factor
  - Feedstock-specific stoichiometric AFR for flue loss calculation
  - Temperature effect on flue gas loss
  - CHP split: electrical and thermal outputs

References:
    Obernberger, I. & Thek, G. (2008). The Pellet Handbook. Earthscan.
    Jenkins, B.M. et al. (1998). Prog. Energy Combust. Sci. 24, 47-81.
    EN 303-5:2012. Heating boilers — Solid fuel heating boilers.
"""

import numpy as np

_H_FG = 2442.0   # kJ/kg (latent heat of vaporisation at 25 degC)


class BiomassCombustionCHPF1b:
    """Biomass CHP with feedstock-specific LHV, moisture correction, and part-load curve."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_rated        = u["Q_rated_kw"]["value"]
        self.eta_el_ref     = u["eta_electrical_ref"]["value"]
        self.eta_th_ref     = u["eta_thermal_ref"]["value"]
        self.PLR_min        = u["PLR_min"]["value"]
        self.a0 = u["a0"]["value"]; self.a1 = u["a1"]["value"]; self.a2 = u["a2"]["value"]
        self.excess_air     = u["excess_air_ratio"]["value"]
        self.stoich_afr_ref = u["stoich_afr"]["value"]
        self.cp_flue        = u["flue_gas_cp"]["value"]
        self.T_flue_full    = u["T_flue_full_degC"]["value"]
        self.T_amb          = u["T_ambient_degC"]["value"]
        self.standby_frac   = u["standby_loss_fraction"]["value"]
        self.cycling_frac   = u["cycling_loss_fraction"]["value"]
        self.feedstock_db   = params["feedstock_db"]

    def _get_feedstock(self, name):
        if name not in self.feedstock_db:
            raise ValueError(f"Unknown feedstock: {name}. Available: {list(self.feedstock_db.keys())}")
        return self.feedstock_db[name]

    def lhv_effective(self, feedstock_type, moisture_fraction):
        """
        LHV_eff [kJ/kg wet fuel] = LHV_dry*(1-M) - h_fg*M
        (Obernberger & Thek 2008; EN303-5)
        """
        fs = self._get_feedstock(feedstock_type)
        M = np.clip(float(moisture_fraction), 0.0, 0.95)
        LHV_dry = fs["LHV_dry_MJ_kg"] * 1000.0  # kJ/kg
        LHV_eff = LHV_dry * (1.0 - M) - _H_FG * M
        return max(0.0, LHV_eff), LHV_dry

    def efficiency_boiler(self, PLR, moisture_fraction, feedstock_type):
        """
        Overall boiler efficiency = (a0 + a1*PLR + a2*PLR^2) * moisture_factor
        moisture_factor = LHV_eff / LHV_dry
        """
        PLR_eff = np.clip(np.asarray(PLR, dtype=float), self.PLR_min, 1.0)
        eta_base = self.a0 + self.a1 * PLR_eff + self.a2 * PLR_eff**2
        LHV_eff, LHV_dry = self.lhv_effective(feedstock_type, moisture_fraction)
        mf = LHV_eff / LHV_dry if LHV_dry > 0 else 0.0
        return np.clip(eta_base * mf, 0.0, 1.0)

    def flue_gas_loss_kw(self, PLR, feedstock_type, moisture_fraction):
        """Flue gas sensible heat loss [kW]."""
        fs = self._get_feedstock(feedstock_type)
        PLR_eff = np.clip(np.asarray(PLR, dtype=float), self.PLR_min, 1.0)
        LHV_eff, _ = self.lhv_effective(feedstock_type, moisture_fraction)
        M = np.clip(float(moisture_fraction), 0.0, 0.95)

        # Fuel input [kW / kJ/kg = kg/s]
        eta = self.efficiency_boiler(PLR, moisture_fraction, feedstock_type)
        Q_out = PLR_eff * self.Q_rated
        Q_fuel = np.where(eta > 0.01, Q_out / eta, Q_out / 0.01)
        m_fuel_kg_s = Q_fuel / max(LHV_eff, 1.0)  # kg/s

        stoich_afr = fs["stoich_afr"]
        m_flue = m_fuel_kg_s * ((1.0 - M) * (1.0 + self.excess_air * stoich_afr) + M)
        T_flue = self.T_amb + (self.T_flue_full - self.T_amb) * (0.35 + 0.65 * PLR_eff)
        return m_flue * self.cp_flue * (T_flue - self.T_amb)

    def predict(self, feedstock_type, PLR, moisture_fraction=None,
                T_ambient_degC=None):
        """
        Predict CHP outputs.

        Args:
            feedstock_type  : str from feedstock_db
            PLR             : part-load ratio [0.2–1.0]
            moisture_fraction: wet-basis [0–0.6] (default = feedstock reference)
            T_ambient_degC  : ambient temperature [degC] (default 15)

        Returns:
            dict: eta_boiler, P_electrical_kw, Q_thermal_kw, fuel_input_kw,
                  flue_loss_kw, standby_loss_kw, cycling_loss_kw,
                  LHV_eff_MJ_kg, moisture_lhv_factor
        """
        fs = self._get_feedstock(feedstock_type)
        if moisture_fraction is None:
            moisture_fraction = fs["moisture_ref"]
        if T_ambient_degC is not None:
            self.T_amb = float(T_ambient_degC)

        PLR_arr = np.asarray(PLR, dtype=float)
        PLR_eff = np.clip(PLR_arr, self.PLR_min, 1.0)

        eta = self.efficiency_boiler(PLR_eff, moisture_fraction, feedstock_type)
        LHV_eff, LHV_dry = self.lhv_effective(feedstock_type, moisture_fraction)
        mf = LHV_eff / LHV_dry if LHV_dry > 0 else 0.0

        Q_out = PLR_eff * self.Q_rated
        Q_fuel = np.where(eta > 0.01, Q_out / eta, 0.0)

        # CHP split: electrical and thermal
        P_el = Q_out * self.eta_el_ref * mf
        Q_th = Q_out * self.eta_th_ref * mf

        flue = self.flue_gas_loss_kw(PLR_eff, feedstock_type, moisture_fraction)
        standby = np.full_like(PLR_arr, self.standby_frac * self.Q_rated)
        cycling = self.cycling_frac * self.Q_rated * (1.0 - PLR_eff)

        return {
            "eta_boiler":          eta,
            "P_electrical_kw":     P_el,
            "Q_thermal_kw":        Q_th,
            "fuel_input_kw":       Q_fuel,
            "flue_loss_kw":        flue,
            "standby_loss_kw":     standby,
            "cycling_loss_kw":     cycling,
            "LHV_eff_MJ_kg":       LHV_eff / 1000.0,
            "moisture_lhv_factor": mf,
        }
