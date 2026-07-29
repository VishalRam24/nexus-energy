"""F0a empirical BMP lookup for EC141 thermophilic anaerobic digester.

Black-box lookup: per-feedstock BMP (L CH4 / kg VS) x temperature de-rating
curve (np.interp). NumPy only, no ODEs.

Data source: Angelidaki et al. (2009); Mata-Alvarez et al. (2014); Labatut
et al. (2011) — BMP values reused from the EC141 F1b feedstock database.
"""
import numpy as np


class BMPTable:
    def __init__(self, params):
        u = params["rated"]
        self.ch4_base = u["methane_content_base"]["value"]
        self.lhv = u["LHV_methane_kwh_m3"]["value"]
        self.T_opt = u["T_optimum_c"]["value"]
        ft = params["feedstock_table"]
        self.feedstocks = ft["feedstocks"]
        self.bmp = dict(zip(ft["feedstocks"], ft["BMP_L_CH4_kgVS"]))
        self.ch4_frac = dict(zip(ft["feedstocks"], ft["methane_content"]))
        tc = params["temperature_curve"]
        self.T_bp = np.asarray(tc["T_degC"], float)
        self.mult_bp = np.asarray(tc["yield_multiplier"], float)

    def temp_multiplier(self, T_degC):
        return float(np.interp(T_degC, self.T_bp, self.mult_bp))

    def bmp_yield(self, feedstock, T_degC=55.0):
        """BMP (L CH4 / kg VS) at temperature."""
        base = self.bmp.get(feedstock, 300.0)
        return base * self.temp_multiplier(T_degC)

    def energy_yield(self, feedstock, T_degC=55.0):
        """Specific energy (kWh / kg VS) = BMP[L]/1000 * LHV."""
        return self.bmp_yield(feedstock, T_degC) / 1000.0 * self.lhv
