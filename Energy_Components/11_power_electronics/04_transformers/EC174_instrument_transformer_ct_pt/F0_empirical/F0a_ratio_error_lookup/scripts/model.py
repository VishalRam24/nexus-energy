"""F0a empirical 1-D lookup model for EC174 Instrument Transformer (CT/PT).

F0 = simplest fidelity: an empirical lookup / rating curve, NumPy only.

A 1-D np.interp lookup over a tabulated breakpoint array for the device's
primary grid metric (loss fraction vs loading, ratio error vs burden, or output
vs demand), plus rated values reused from the F1 datasheet.

Source (reused from F1a): IEC 61869-2:2012 (CT); IEC 61869-3:2011 (PT); IEEE C57.13-2016 (reused from F1a)
"""
import numpy as np


class LookupCurve:
    """Generic 1-D breakpoint lookup for a grid component."""

    def __init__(self, x_breakpoints, y_breakpoints):
        self.x = np.asarray(x_breakpoints, dtype=float)
        self.y = np.asarray(y_breakpoints, dtype=float)

    def lookup(self, x):
        """Interpolated value at x (np.interp, clamped to table ends)."""
        xc = np.clip(np.asarray(x, dtype=float), self.x[0], self.x[-1])
        return np.interp(xc, self.x, self.y)
