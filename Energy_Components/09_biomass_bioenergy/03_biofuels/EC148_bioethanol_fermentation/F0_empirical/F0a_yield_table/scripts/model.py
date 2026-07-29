"""F0a empirical ethanol-yield lookup for EC148 bioethanol fermentation.

Black-box model: per-feedstock ethanol yield (L EtOH / tonne feed) x a
fermentation-temperature multiplier (np.interp, peak ~32 C). NumPy only.

Data source: Chandel et al. (2011); Gay-Lussac stoichiometry; S. cerevisiae
data — yields reused from the EC148 F1b parameter set.
"""
import numpy as np


class EthanolYieldTable:
    def __init__(self, params):
        r = params["rated"]
        self.yeast_eff = r["yeast_efficiency"]["value"]
        self.T_opt = r["T_optimum_degC"]["value"]
        self.lhv = r["LHV_ethanol_MJ_L"]["value"]
        ft = params["feedstock_table"]
        self.feedstocks = ft["feedstocks"]
        self.yield_opt = dict(zip(ft["feedstocks"], ft["ethanol_L_per_tonne"]))
        tc = params["temperature_curve"]
        self.T_bp = np.asarray(tc["T_degC"], float)
        self.mult_bp = np.asarray(tc["yield_multiplier"], float)

    def temp_multiplier(self, T_degC):
        return float(np.interp(T_degC, self.T_bp, self.mult_bp))

    def ethanol_yield(self, feedstock, T_degC=32.0):
        """Ethanol yield (L / tonne feed) at temperature."""
        base = self.yield_opt.get(feedstock, 200.0)
        return base * self.temp_multiplier(T_degC)

    def energy_yield_MJ_tonne(self, feedstock, T_degC=32.0):
        return self.ethanol_yield(feedstock, T_degC) * self.lhv
