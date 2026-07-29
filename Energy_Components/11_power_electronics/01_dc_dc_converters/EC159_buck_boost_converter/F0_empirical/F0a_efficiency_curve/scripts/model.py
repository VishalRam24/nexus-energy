"""F0a empirical efficiency-vs-load lookup model for EC159 Buck-Boost Converter.

Fidelity F0a: simplest faithful model. A part-load efficiency curve stored as
tabulated (load_fraction, efficiency) breakpoints, evaluated by 1-D linear
interpolation (numpy.interp). NumPy only -- no scipy, no ODEs, no AI.

Data source: Erickson & Maksimovic (2001), Fundamentals of Power Electronics, 2nd ed.; Kazimierczuk (2015), PWM DC-DC Power Converters, Wiley
Curve shape: efficiency rises from 0 at no load, peaks near mid-load (switching
and fixed losses dominate at light load, conduction/copper losses at full load).
"""
import json
import os
import numpy as np


class EfficiencyCurveModel:
    def __init__(self, params_path=None):
        if params_path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            params_path = os.path.join(here, "..", "data", "parameters.json")
        with open(params_path) as f:
            p = json.load(f)
        m = p["model"]
        self.component = p["component"]
        self.load_fraction = np.asarray(m["load_fraction"]["value"], dtype=float)
        self.efficiency = np.asarray(m["efficiency"]["value"], dtype=float)
        self.p_rated = float(m["p_rated"]["value"])
        self.rated_efficiency = float(m["rated_efficiency"]["value"])
        self.source = m["source"]

    def efficiency_at(self, load_fraction):
        """Interpolated efficiency at a given load fraction (clamped to [0, max])."""
        lf = np.clip(np.asarray(load_fraction, dtype=float), 0.0,
                     float(self.load_fraction[-1]))
        return np.interp(lf, self.load_fraction, self.efficiency)

    def power_out(self, load_fraction):
        """Delivered output power (W or VA) at a given load fraction."""
        lf = np.asarray(load_fraction, dtype=float)
        return lf * self.p_rated

    def losses(self, load_fraction):
        """Losses (W) = P_out * (1/eta - 1); 0 when no load."""
        lf = np.asarray(load_fraction, dtype=float)
        eta = self.efficiency_at(lf)
        p_out = lf * self.p_rated
        with np.errstate(divide="ignore", invalid="ignore"):
            loss = np.where(eta > 0.0, p_out * (1.0 / eta - 1.0), 0.0)
        return loss
