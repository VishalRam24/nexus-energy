"""F0a empirical cold-gas-efficiency curve for EC143 biomass gasifier.

Black-box model: CGE vs equivalence ratio (np.interp curve, peaks at ER=0.25),
with a mild per-feedstock scaling by HHV and a linear moisture penalty.
NumPy only.

Data source: Basu (2010) Biomass Gasification — curve and feedstock HHV reused
from the EC143 F1b parameter set.
"""
import numpy as np


class CGECurve:
    def __init__(self, params):
        u = params["rated"]
        self.ER_design = u["ER_design"]["value"]
        self.cge_design = u["CGE_design"]["value"]
        self.lhv_syngas = u["LHV_syngas_MJ_Nm3_design"]["value"]
        ec = params["er_curve"]
        self.ER_bp = np.asarray(ec["ER"], float)
        self.CGE_bp = np.asarray(ec["CGE"], float)
        fh = params["feedstock_hhv"]
        self.feedstocks = fh["feedstocks"]
        self.hhv = dict(zip(fh["feedstocks"], fh["HHV_MJ_kg"]))

    def cge_vs_er(self, ER):
        """Cold gas efficiency at equivalence ratio ER (clamped)."""
        return float(np.interp(ER, self.ER_bp, self.CGE_bp))

    def fuel_factor(self, feedstock):
        """Mild fuel-quality scaling, normalised so wood (HHV 20) -> 1.0."""
        return self.hhv.get(feedstock, 20.0) / 20.0

    def cge(self, feedstock="wood", ER=0.25, moisture=0.0):
        """Net CGE: curve x fuel factor x moisture penalty (-1.2 per moisture frac)."""
        base = self.cge_vs_er(ER) * self.fuel_factor(feedstock)
        penalty = max(0.0, 1.0 - 1.2 * moisture)
        return base * penalty
