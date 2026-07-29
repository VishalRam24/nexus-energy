"""
EC142 -- Biogas Upgrading / Biomethane -- F1b Feedstock Variation Model

Extends F1a (single feedstock, fixed composition) with:
  - Feedstock-dependent raw biogas composition (CH4, CO2, H2S, H2O, N2)
  - Co-digestion blending of biogas composition
  - Moisture content effect on raw biogas energy density (moisture-LHV coupling)
  - Upgrading energy penalty as function of CO2 fraction (higher CO2 → more energy)
  - Biomethane quality metrics vs specification
  - H2S removal sizing

Moisture-LHV coupling:
    LHV_eff = LHV_dry * (1 - M) - h_fg * M
    Applied to raw biogas energy content before upgrading credit.

Upgrading energy (PSA/membrane empirical):
    E_upg = E_base * (1 + 0.5 * (CO2_frac - 0.38))
    (higher CO2 fraction requires more adsorbent regeneration)

References:
    Bauer, F. et al. (2013). Bioresource Technology 148, 597-606.
    Ryckebosch, E. et al. (2011). Biomass & Bioenergy 35, 1633-1645.
    IEA Bioenergy Task 37 (2019). Upgrading Biogas to Biomethane.
"""

import numpy as np

_H_FG = 2442.0   # kJ/kg
_LHV_CH4_kwh_m3 = 9.97


class BiogasUpgradingF1b:
    """Biogas upgrading model with feedstock-specific composition."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.feedstock_db   = params["feedstock_db"]
        self.spec           = params["biomethane_spec"]
        self.methane_slip   = u["methane_slip_pct"]["value"] / 100.0
        self.upg_energy_base = u["upgrading_energy_kwh_m3_biogas"]["value"]
        self.H2S_rem_eff    = u["H2S_removal_eff"]["value"]
        self.H2O_rem_eff    = u["water_removal_eff"]["value"]

    def _parse_feedstock(self, feedstock_type):
        if isinstance(feedstock_type, str):
            if feedstock_type not in self.feedstock_db:
                raise ValueError(f"Unknown feedstock: {feedstock_type}. "
                                 f"Available: {list(self.feedstock_db.keys())}")
            return {k: v for k, v in self.feedstock_db[feedstock_type].items()}
        elif isinstance(feedstock_type, dict):
            total = sum(feedstock_type.values())
            blend = {k: 0.0 for k in ["CH4_pct","CO2_pct","H2S_ppm","H2O_pct",
                                       "N2_pct","biogas_yield_m3_tVS","C_pct","N_pct"]}
            for name, frac in feedstock_type.items():
                if name not in self.feedstock_db:
                    raise ValueError(f"Unknown feedstock: {name}")
                for k in blend:
                    blend[k] += (frac / total) * self.feedstock_db[name][k]
            return blend
        raise TypeError("feedstock_type must be str or dict")

    def moisture_lhv_factor(self, moisture_fraction):
        """LHV_eff / LHV_dry for wet biomass (EC087 convention)."""
        M = np.clip(float(moisture_fraction), 0.0, 0.95)
        LHV_dry = 18000.0
        LHV_eff = LHV_dry * (1.0 - M) - _H_FG * M
        return max(0.0, LHV_eff / LHV_dry)

    def upgrading_energy(self, CO2_frac):
        """
        Electricity consumption for upgrading [kWh/m3 raw biogas].
        Scales with CO2 fraction: higher CO2 → more compression/adsorption work.
        Reference: 0.25 kWh/m3 at CO2=38% (cattle manure baseline).
        """
        return self.upg_energy_base * (1.0 + 0.5 * (float(CO2_frac) - 0.38))

    def predict(self, feedstock_type, biogas_flow_m3_h, moisture_fraction=0.0,
                temperature_degC=20.0):
        """
        Predict biomethane production and energy balance.

        Args:
            feedstock_type     : str or dict of blends
            biogas_flow_m3_h   : raw biogas flow [m3/h]
            moisture_fraction  : wet-basis moisture of feedstock [0-1]
            temperature_degC   : operating temperature [degC]

        Returns:
            dict: biomethane_flow_m3_h, methane_recovery_pct, biomethane_CH4_pct,
                  CO2_removal_pct, H2S_product_ppm, upgrading_energy_kwh_h,
                  net_energy_kwh_h, meets_spec, moisture_lhv_factor
        """
        comp = self._parse_feedstock(feedstock_type)
        Q_raw = float(biogas_flow_m3_h)

        CH4_frac  = comp["CH4_pct"] / 100.0
        CO2_frac  = comp["CO2_pct"] / 100.0
        H2S_ppm   = comp["H2S_ppm"]
        H2O_frac  = comp["H2O_pct"] / 100.0
        N2_frac   = comp["N2_pct"] / 100.0

        # Moisture-LHV coupling applied to raw biogas energy
        f_moisture = self.moisture_lhv_factor(moisture_fraction)

        # CH4 entering upgrader
        Q_CH4_in = Q_raw * CH4_frac  # m3/h

        # After upgrading: methane retained (minus slip)
        Q_CH4_out = Q_CH4_in * (1.0 - self.methane_slip)

        # Biomethane flow (CH4 + residual CO2 + trace N2)
        # Target: 97% CH4 in product; residual CO2 = 3%
        # Product stream: Q_CH4_out / 0.97
        CH4_product_frac = 0.97  # targeted biomethane quality
        Q_biomethane = Q_CH4_out / CH4_product_frac

        # Residual CO2 in product
        CO2_product_frac = 1.0 - CH4_product_frac - N2_frac * 0.1
        CO2_product_frac = max(0.0, min(CO2_product_frac, 0.04))

        # H2S in product
        H2S_product_ppm = H2S_ppm * (1.0 - self.H2S_rem_eff)

        # CO2 removed
        CO2_removal = max(0.0, 1.0 - CO2_product_frac / CO2_frac) if CO2_frac > 0 else 1.0

        # Methane recovery
        methane_recovery = (Q_CH4_out / Q_CH4_in * 100.0) if Q_CH4_in > 0 else 0.0

        # Upgrading energy
        E_upg = self.upgrading_energy(CO2_frac) * Q_raw  # kWh/h

        # Gross energy in biomethane
        E_gross = Q_biomethane * _LHV_CH4_kwh_m3 * f_moisture  # kWh/h

        # Net energy (gross - upgrading parasitic)
        E_net = E_gross - E_upg

        # Spec compliance
        meets_spec = (
            CH4_product_frac * 100.0 >= self.spec["CH4_min_pct"] and
            CO2_product_frac * 100.0 <= self.spec["CO2_max_pct"] and
            H2S_product_ppm <= self.spec["H2S_max_ppm"]
        )

        return {
            "biomethane_flow_m3_h":  float(Q_biomethane),
            "methane_recovery_pct":  float(methane_recovery),
            "biomethane_CH4_pct":    float(CH4_product_frac * 100.0),
            "CO2_removal_pct":       float(CO2_removal * 100.0),
            "H2S_product_ppm":       float(H2S_product_ppm),
            "upgrading_energy_kwh_h":float(E_upg),
            "net_energy_kwh_h":      float(E_net),
            "meets_spec":            bool(meets_spec),
            "moisture_lhv_factor":   float(f_moisture),
        }
