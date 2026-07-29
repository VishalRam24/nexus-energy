"""F0a empirical upgrading lookup for EC142 biogas upgrading / biomethane.

Black-box PSA performance lookup: per-feedstock raw CH4 content drives the
biomethane volume produced, with constant CH4 recovery (1 - slip) and constant
parasitic upgrading energy. NumPy only.

Data source: Bauer et al. (2013); Ryckebosch et al. (2011) — values reused
from the EC142 F1b parameter set.
"""
import numpy as np


class UpgradingLookup:
    def __init__(self, params):
        u = params["rated"]
        self.slip = u["methane_slip_pct"]["value"] / 100.0
        self.recovery = 1.0 - self.slip
        self.upg_energy = u["upgrading_energy_kwh_m3_biogas"]["value"]
        self.product_purity = u["product_CH4_pct"]["value"]
        self.lhv = u["LHV_methane_kwh_m3"]["value"]
        ft = params["feedstock_table"]
        self.feedstocks = ft["feedstocks"]
        self.raw_ch4 = dict(zip(ft["feedstocks"], ft["raw_CH4_pct"]))
        # interp breakpoints for arbitrary raw CH4 input
        self._raw_bp = np.asarray(sorted(ft["raw_CH4_pct"]), float)

    def raw_ch4_pct(self, feedstock):
        return self.raw_ch4.get(feedstock, 60.0)

    def biomethane_per_biogas(self, raw_ch4_pct):
        """Nm3 biomethane (at product purity) per Nm3 raw biogas."""
        ch4_in = raw_ch4_pct / 100.0
        ch4_out = ch4_in * self.recovery
        return ch4_out / (self.product_purity / 100.0)

    def parasitic_fraction(self, raw_ch4_pct):
        """Parasitic energy as a fraction of biomethane LHV energy."""
        bm = self.biomethane_per_biogas(raw_ch4_pct)
        energy_out = bm * self.lhv  # kWh per m3 raw biogas
        return self.upg_energy / energy_out
