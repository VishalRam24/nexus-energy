"""EC221 MHD Generator F0a - empirical power-density vs velocity lookup.

Electrical power density of a Faraday MHD channel tabulated against plasma flow
velocity. Breakpoints from P_density = sigma*u^2*B^2*K*(1-K) at sigma=10 S/m,
B=5 T, K=0.5 (maximum-power load factor). Total power is the density scaled by
the channel volume.

Data source: Rosa, R.J. (1987) Magnetohydrodynamic Energy Conversion;
Way, S. et al. (1979) AIAA.
NumPy only - no scipy, no ODEs, no AI.
"""
import json
import os
import numpy as np


class MHDPowerCurve:
    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        with open(params_path) as f:
            p = json.load(f)
        self.volume = p["channel_volume_m3"]["value"]
        self._u = np.asarray(p["lookup"]["u_m_s"]["value"], dtype=float)
        self._pd = np.asarray(p["lookup"]["power_density_W_m3"]["value"], dtype=float)

    def power_density(self, u_m_s):
        """Interpolated electrical power density (W/m^3) vs flow velocity."""
        return float(np.interp(u_m_s, self._u, self._pd))

    def power(self, u_m_s):
        """Total electrical power (W) over the channel volume."""
        return self.power_density(u_m_s) * self.volume

    def breakpoints(self):
        return self._u.copy(), self._pd.copy()
