"""EC217 TEC F0a - empirical COP-vs-deltaT lookup.

Coefficient of performance of a Bi2Te3 Peltier cooler tabulated against the
temperature lift deltaT = Th - Tc (hot side fixed at 300 K). Breakpoints follow
the standard TEC max-COP relation
    COP = Tc/(Th-Tc) * (sqrt(1+ZT) - Th/Tc) / (sqrt(1+ZT) + 1)
with ZT = 0.7 (Bi2Te3). COP falls toward zero as deltaT approaches dT_max.

Data source: Rowe (2006) Thermoelectrics Handbook; Goldsmid (2010).
NumPy only - no scipy, no ODEs, no AI.
"""
import json
import os
import numpy as np


class TECCopMap:
    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        with open(params_path) as f:
            p = json.load(f)
        self.ZT = p["ZT"]["value"]
        self.Th_ref = p["Th_ref"]["value"]
        self.dT_max = p["dT_max"]["value"]
        self._dt = np.asarray(p["lookup"]["delta_T_K"]["value"], dtype=float)
        self._cop = np.asarray(p["lookup"]["COP"]["value"], dtype=float)

    def cop(self, delta_T_K):
        """Interpolated coefficient of performance vs temperature lift."""
        return float(np.interp(delta_T_K, self._dt, self._cop))

    def breakpoints(self):
        return self._dt.copy(), self._cop.copy()
