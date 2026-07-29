"""EC216 TEG F0a - empirical efficiency-vs-deltaT lookup.

Conversion efficiency of a Bi2Te3 thermoelectric generator module tabulated
against hot-side temperature (cold side fixed at 30 C). The breakpoints follow
the standard TEG efficiency relation
    eta = eta_carnot * (sqrt(1+ZT)-1) / (sqrt(1+ZT) + Tc/Th)
with ZT = 1.0 (Bi2Te3).

Data source: Rowe (2006) Thermoelectrics Handbook;
Snyder & Toberer (2008), Nature Materials, 7, 105-114.
NumPy only - no scipy, no ODEs, no AI.
"""
import json
import os
import numpy as np


class TEGEfficiencyCurve:
    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        with open(params_path) as f:
            p = json.load(f)
        self.ZT = p["ZT"]["value"]
        self.T_cold_ref = p["T_cold_ref"]["value"]
        self._th = np.asarray(p["lookup"]["T_hot_degC"]["value"], dtype=float)
        self._eff = np.asarray(p["lookup"]["efficiency"]["value"], dtype=float)

    def efficiency(self, T_hot_degC):
        """Interpolated module conversion efficiency (fraction) vs hot-side temp."""
        return float(np.interp(T_hot_degC, self._th, self._eff))

    def breakpoints(self):
        return self._th.copy(), self._eff.copy()
