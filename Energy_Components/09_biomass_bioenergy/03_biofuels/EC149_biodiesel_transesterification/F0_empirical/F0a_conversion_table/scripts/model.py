"""F0a empirical conversion model for EC149 biodiesel transesterification.

Black-box model: conversion = base_conversion x FFA-penalty(np.interp) x
temperature-multiplier(np.interp). Biodiesel mass = oil_content x conversion.
NumPy only.

Data source: Freedman (1984); Meher (2006); Rashid (2008) — values reused from
the EC149 F1b parameter set.
"""
import numpy as np


class ConversionTable:
    def __init__(self, params):
        r = params["rated"]
        self.base_conv = r["base_conversion"]["value"]
        self.T_opt = r["T_optimum_degC"]["value"]
        self.lhv = r["LHV_biodiesel_MJ_kg"]["value"]
        fp = params["ffa_penalty_curve"]
        self.ffa_bp = np.asarray(fp["ffa_pct"], float)
        self.ffa_mult_bp = np.asarray(fp["conversion_multiplier"], float)
        ft = params["feedstock_table"]
        self.feedstocks = ft["feedstocks"]
        self.oil = dict(zip(ft["feedstocks"], ft["oil_content_frac"]))
        self.ffa = dict(zip(ft["feedstocks"], ft["ffa_pct"]))
        tc = params["temperature_curve"]
        self.T_bp = np.asarray(tc["T_degC"], float)
        self.T_mult_bp = np.asarray(tc["temp_multiplier"], float)

    def ffa_multiplier(self, ffa_pct):
        return float(np.interp(ffa_pct, self.ffa_bp, self.ffa_mult_bp))

    def temp_multiplier(self, T_degC):
        return float(np.interp(T_degC, self.T_bp, self.T_mult_bp))

    def conversion(self, feedstock=None, ffa_pct=None, T_degC=60.0):
        if ffa_pct is None:
            ffa_pct = self.ffa.get(feedstock, 1.0)
        return self.base_conv * self.ffa_multiplier(ffa_pct) * self.temp_multiplier(T_degC)

    def biodiesel_frac(self, feedstock, T_degC=60.0):
        """Biodiesel mass fraction of feed = oil content x conversion."""
        return self.oil.get(feedstock, 0.2) * self.conversion(feedstock=feedstock, T_degC=T_degC)
