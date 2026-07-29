"""F0a empirical ASF alpha curve for EC150 Fischer-Tropsch synthesis (BtL).

Black-box model: chain-growth probability alpha and CO conversion vs reactor
temperature (np.interp). Diesel-cut (C10-C20) selectivity is computed from the
analytic Anderson-Schulz-Flory mass distribution w_n = n*(1-a)^2*a^(n-1).
NumPy only.

Data source: Dry (2002); van der Laan & Beenackers (1999) — alpha and CO
conversion reused from the EC150 F1b parameter set.
"""
import numpy as np


class AlphaCurve:
    def __init__(self, params):
        r = params["rated"]
        self.T_ref = r["T_ref_degC"]["value"]
        self.alpha_ref = r["alpha_ref"]["value"]
        ac = params["alpha_curve"]
        self.T_bp = np.asarray(ac["T_degC"], float)
        self.alpha_bp = np.asarray(ac["alpha"], float)
        cc = params["co_conversion_curve"]
        self.coT_bp = np.asarray(cc["T_degC"], float)
        self.co_bp = np.asarray(cc["CO_conversion"], float)
        d = params["diesel_carbon_range"]
        self.c_min, self.c_max = int(d["C_min"]), int(d["C_max"])

    def alpha(self, T_degC):
        return float(np.interp(T_degC, self.T_bp, self.alpha_bp))

    def co_conversion(self, T_degC):
        return float(np.interp(T_degC, self.coT_bp, self.co_bp))

    def diesel_selectivity(self, T_degC):
        """ASF mass fraction in C10-C20 cut at the temperature's alpha."""
        a = self.alpha(T_degC)
        n = np.arange(1, 100)
        w = n * (1.0 - a) ** 2 * a ** (n - 1)  # ASF mass distribution
        mask = (n >= self.c_min) & (n <= self.c_max)
        return float(w[mask].sum())
