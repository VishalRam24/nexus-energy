"""F0a empirical methane-yield lookup for EC140 mesophilic anaerobic digester.

A black-box lookup model: per-feedstock specific methane yield (m3 CH4 / kg VS)
multiplied by a temperature de-rating curve (np.interp). NO ODEs, NumPy only.

Data source: Buswell & Mueller (1952); Batstone et al. (2002) IWA ADM1 simplified;
Labatut et al. (2011) — reused from the EC140 F1a / EC141 F1b parameter sets.
"""
import numpy as np


class YieldTable:
    def __init__(self, params):
        u = params["rated"]
        self.Y_max = u["Y_max_m3_kgVS"]["value"]
        self.lhv = u["LHV_methane_kwh_m3"]["value"]
        self.T_opt = u["T_optimal_c"]["value"]
        ft = params["feedstock_table"]
        self.feedstocks = ft["feedstocks"]
        self.ch4_yield = dict(zip(ft["feedstocks"], ft["CH4_yield_m3_kgVS"]))
        self.ch4_frac = dict(zip(ft["feedstocks"], ft["methane_fraction"]))
        tc = params["temperature_curve"]
        self.T_bp = np.asarray(tc["T_degC"], float)
        self.mult_bp = np.asarray(tc["yield_multiplier"], float)

    def temp_multiplier(self, T_degC):
        """Yield de-rating factor at temperature T (clamped to table ends)."""
        return float(np.interp(T_degC, self.T_bp, self.mult_bp))

    def methane_yield(self, feedstock, T_degC=37.0):
        """Specific CH4 yield (m3 CH4 / kg VS) for feedstock at temperature."""
        base = self.ch4_yield.get(feedstock, self.Y_max)
        return base * self.temp_multiplier(T_degC)

    def energy_yield(self, feedstock, T_degC=37.0):
        """Specific energy yield (kWh / kg VS)."""
        return self.methane_yield(feedstock, T_degC) * self.lhv
