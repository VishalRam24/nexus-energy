"""F0a empirical mass-yield / energy-densification table for EC146 torrefaction.

Black-box model: solid mass yield and energy densification ratio (EDR) vs
temperature (np.interp). Energy yield = mass_yield * EDR. NumPy only.

Data source: Bach & Skreiberg (2016); Bates & Ghoniem (2012) — curves reused
from the EC146 F1b parameter set.
"""
import numpy as np


class MassYieldTable:
    def __init__(self, params):
        r = params["rated"]
        self.T_ref = r["T_ref_degC"]["value"]
        self.lhv_raw = r["LHV_dry_MJ_kg"]["value"]
        yc = params["yield_curves"]
        self.T_bp = np.asarray(yc["T_degC"], float)
        self.my_bp = np.asarray(yc["mass_yield"], float)
        self.edr_bp = np.asarray(yc["EDR"], float)

    def mass_yield(self, T_degC):
        return float(np.interp(T_degC, self.T_bp, self.my_bp))

    def edr(self, T_degC):
        return float(np.interp(T_degC, self.T_bp, self.edr_bp))

    def energy_yield(self, T_degC):
        """Fraction of feed energy retained in solid = mass_yield * EDR."""
        return self.mass_yield(T_degC) * self.edr(T_degC)

    def torrefied_lhv(self, T_degC):
        return self.lhv_raw * self.edr(T_degC)
