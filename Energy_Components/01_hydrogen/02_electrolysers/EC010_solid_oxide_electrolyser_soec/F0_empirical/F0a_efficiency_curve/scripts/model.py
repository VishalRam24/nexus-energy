"""F0a empirical lookup model for EC010 — Solid Oxide Electrolyser (SOEC).

Fidelity F0 (empirical): a 1-D NumPy lookup / interpolation over tabulated
breakpoints. No ODEs, no scipy, no AI — pure NumPy. Numbers reuse the
component's F1 parameters / literature.

Source: Ni et al. (2007), Chem. Eng. Tech., 29(6), 636-642 (F1a params reused)
Metric: h2_production_efficiency_LHV
SOEC very high electrical efficiency (~80-95%) at high T; thermoneutral V=1.285 V. Excludes heat input.
"""
import numpy as np


class EfficiencyCurve:
    """1-D empirical lookup: y = f(x) via np.interp over tabulated breakpoints."""

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        if self.x.ndim != 1 or self.x.shape != self.y.shape:
            raise ValueError("x and y must be 1-D arrays of equal length")
        # np.interp requires increasing x
        order = np.argsort(self.x)
        self.x = self.x[order]
        self.y = self.y[order]

    def lookup(self, xq):
        """Interpolate y at query x (scalar or array). Clamps to table endpoints."""
        xq = np.asarray(xq, dtype=float)
        return np.interp(xq, self.x, self.y)

    @property
    def x_min(self):
        return float(self.x[0])

    @property
    def x_max(self):
        return float(self.x[-1])
