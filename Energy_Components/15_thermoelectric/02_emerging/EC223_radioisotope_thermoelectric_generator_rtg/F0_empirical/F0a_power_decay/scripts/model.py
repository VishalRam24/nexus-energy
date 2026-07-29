"""EC223 RTG F0a - empirical electrical-power vs time decay lookup.

Net electrical power of a Pu-238 GPHS-RTG tabulated against mission time. The
breakpoints combine isotope decay heat
    P_thermal(t) = P_thermal_0 * exp(-ln2*t/t_half)
with a linearly degrading thermoelectric efficiency eta_teg(t), giving
P_elec(t) = P_thermal(t)*eta_teg(t). A 1-D interpolation reproduces the
multi-decade power decline used to size deep-space missions.

Data source: Bennett (2006) Acta Astronautica; El-Genk & Saber (2005) Energy
Convers. Mgmt.; NASA GPHS-RTG spec.
NumPy only - no scipy, no ODEs, no AI.
"""
import json
import os
import numpy as np


class RTGPowerDecay:
    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        with open(params_path) as f:
            p = json.load(f)
        self.t_half = p["t_half_years"]["value"]
        self.P_thermal_0 = p["P_thermal_0_W"]["value"]
        self._t = np.asarray(p["lookup"]["t_years"]["value"], dtype=float)
        self._pw = np.asarray(p["lookup"]["power_W"]["value"], dtype=float)

    def power_W(self, t_years):
        """Interpolated net electrical power (W) vs mission time."""
        return float(np.interp(t_years, self._t, self._pw))

    def breakpoints(self):
        return self._t.copy(), self._pw.copy()
