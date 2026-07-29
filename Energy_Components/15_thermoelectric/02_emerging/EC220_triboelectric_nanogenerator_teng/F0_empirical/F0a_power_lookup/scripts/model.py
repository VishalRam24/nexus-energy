"""EC220 TENG F0a - empirical average-power vs frequency lookup.

Average output power of a PTFE-Al contact-separation triboelectric
nanogenerator into its optimal ~10 MOhm load, tabulated against the
contact-separation frequency. Energy per cycle is fixed by surface charge and
gap, so average power scales linearly with frequency.

Data source: Wang, Z.L. et al. (2012) Nano Lett.; Fan et al. (2012) Nano
Energy; Wang et al. (2015) Mater. Today.
NumPy only - no scipy, no ODEs, no AI.
"""
import json
import os
import numpy as np


class TENGPowerLookup:
    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        with open(params_path) as f:
            p = json.load(f)
        self.E_cycle = p["energy_per_cycle_J"]["value"]
        self._f = np.asarray(p["lookup"]["frequency_Hz"]["value"], dtype=float)
        self._pw = np.asarray(p["lookup"]["power_mW"]["value"], dtype=float)

    def power_mW(self, frequency_Hz):
        """Interpolated average output power (mW) vs cycle frequency."""
        return float(np.interp(frequency_Hz, self._f, self._pw))

    def breakpoints(self):
        return self._f.copy(), self._pw.copy()
