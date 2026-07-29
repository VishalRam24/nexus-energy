"""EC219 Piezoelectric Energy Harvester F0a - empirical power vs acceleration lookup.

Harvested electrical power of a PZT-5A bimorph cantilever at mechanical
resonance (100 Hz) into an impedance-matched load, tabulated against base
acceleration. At resonance harvested power scales with acceleration^2; the
table is anchored at 0.5 mW for 1 g (9.81 m/s^2).

Data source: Roundy et al. (2003) Smart Mater. Struct.;
Erturk & Inman (2011) Piezoelectric Energy Harvesting.
NumPy only - no scipy, no ODEs, no AI.
"""
import json
import os
import numpy as np


class PiezoPowerLookup:
    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        with open(params_path) as f:
            p = json.load(f)
        self.f_res = p["resonance_freq_hz"]["value"]
        self._acc = np.asarray(p["lookup"]["acceleration_m_s2"]["value"], dtype=float)
        self._pw = np.asarray(p["lookup"]["power_mW"]["value"], dtype=float)

    def power_mW(self, acceleration_m_s2):
        """Interpolated harvested power (mW) vs base acceleration at resonance."""
        return float(np.interp(acceleration_m_s2, self._acc, self._pw))

    def breakpoints(self):
        return self._acc.copy(), self._pw.copy()
