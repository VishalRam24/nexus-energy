"""EC222 Betavoltaic Cell F0a - empirical power-vs-time decay lookup.

Output electrical power of a Ni-63 betavoltaic cell tabulated against time. The
power follows the isotope decay law P(t)=P0*exp(-ln2*t/t_half) with
P0 = A0*E_beta*eta_capture*eta_conv. A 1-D interpolation over the tabulated
breakpoints reproduces the slow exponential decline over decades.

Data source: Blanovsky (2012) IEEE AERO; Olsen et al. (1993) Nucl. Instrum.
Methods; Sychov et al. (2008) Appl. Radiat. Isot.
NumPy only - no scipy, no ODEs, no AI.
"""
import json
import os
import numpy as np


class BetavoltaicPowerDecay:
    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        with open(params_path) as f:
            p = json.load(f)
        self.P0_nW = p["P0_nW"]["value"]
        self.t_half = p["t_half_years"]["value"]
        self._t = np.asarray(p["lookup"]["t_years"]["value"], dtype=float)
        self._pw = np.asarray(p["lookup"]["power_nW"]["value"], dtype=float)

    def power_nW(self, t_years):
        """Interpolated output power (nW) vs time since manufacture."""
        return float(np.interp(t_years, self._t, self._pw))

    def breakpoints(self):
        return self._t.copy(), self._pw.copy()
