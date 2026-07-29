"""F0a empirical bio-crude yield lookup for EC147 hydrothermal liquefaction.

Black-box model: per-feedstock bio-crude mass yield x temperature de-rating
curve (np.interp, peak at 330 C). NumPy only.

Data source: Peterson et al. (2008); Vardon et al. (2011); Elliott et al.
(2015) — yields reused from the EC147 F1b parameter set.
"""
import numpy as np


class BiocrudeTable:
    def __init__(self, params):
        r = params["rated"]
        self.T_ref = r["T_ref_degC"]["value"]
        self.hhv = r["HHV_biocrude_MJ_kg"]["value"]
        ft = params["feedstock_table"]
        self.feedstocks = ft["feedstocks"]
        self.yield_opt = dict(zip(ft["feedstocks"], ft["biocrude_yield"]))
        tc = params["temperature_curve"]
        self.T_bp = np.asarray(tc["T_degC"], float)
        self.mult_bp = np.asarray(tc["yield_multiplier"], float)

    def temp_multiplier(self, T_degC):
        return float(np.interp(T_degC, self.T_bp, self.mult_bp))

    def biocrude_yield(self, feedstock, T_degC=330.0):
        base = self.yield_opt.get(feedstock, 0.30)
        return base * self.temp_multiplier(T_degC)

    def energy_yield_MJ_kg(self, feedstock, T_degC=330.0):
        """Energy in bio-crude per kg dry feed = yield * HHV."""
        return self.biocrude_yield(feedstock, T_degC) * self.hhv
