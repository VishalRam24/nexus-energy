"""F0a empirical power-curve lookup for EC065 Offshore Fixed-Bottom Wind Turbine (Siemens SWT-3.6-120).

F0 fidelity: a tabulated manufacturer/reference power curve (power vs wind
speed) evaluated by 1-D linear interpolation (np.interp) with cut-in / rated /
cut-out enforcement. Pure NumPy, no scipy/AI.

Data source: Siemens SWT-3.6-120 datasheet; windpowerlib; IEC 61400-12-1
"""
import numpy as np


class WindPowerCurve:
    def __init__(self, wind_speeds, power_kw, cut_in, rated, cut_out,
                 rated_power_kw, motion_penalty=0.0):
        self.v = np.asarray(wind_speeds, dtype=float)
        self.p = np.asarray(power_kw, dtype=float)
        self.cut_in = float(cut_in)
        self.rated = float(rated)
        self.cut_out = float(cut_out)
        self.rated_power_kw = float(rated_power_kw)
        self.motion_penalty = float(motion_penalty)

    def power(self, wind_speed):
        """Return electrical power (kW) for scalar or array wind speed (m/s)."""
        ws = np.asarray(wind_speed, dtype=float)
        p = np.interp(ws, self.v, self.p, left=0.0, right=self.p[-1])
        # enforce cut-in / cut-out (zero outside the operating window)
        p = np.where(ws < self.cut_in, 0.0, p)
        p = np.where(ws > self.cut_out, 0.0, p)
        p = p * (1.0 - self.motion_penalty)
        # never exceed rated
        p = np.minimum(p, self.rated_power_kw)
        return p if p.ndim else float(p)

    def capacity_factor(self, wind_speed):
        return self.power(wind_speed) / self.rated_power_kw
