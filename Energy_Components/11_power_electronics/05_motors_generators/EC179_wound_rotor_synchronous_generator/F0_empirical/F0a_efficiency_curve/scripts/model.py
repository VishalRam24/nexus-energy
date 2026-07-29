"""F0a empirical part-load efficiency-curve lookup for EC179 Wound Rotor Synchronous Generator.

F0 = simplest fidelity: an empirical efficiency curve, NumPy only.

A 1-D part-load efficiency lookup (np.interp over tabulated breakpoints). The
breakpoints come from a two-component (constant + load^2) loss split anchored at
the F1 rated efficiency, so eta(1.0) == eta_rated exactly.

Source (reused from F1a): Boldea (2015), Synchronous Generators, 2nd ed. (reused from F1a)
"""
import numpy as np


class EfficiencyCurve:
    """Generic part-load efficiency lookup for a rotating/static machine."""

    def __init__(self, load_breakpoints, eff_breakpoints, eta_rated):
        self.load = np.asarray(load_breakpoints, dtype=float)
        self.eff = np.asarray(eff_breakpoints, dtype=float)
        self.eta_rated = float(eta_rated)

    def efficiency(self, load_fraction):
        """Efficiency at a fraction of rated load (np.interp, clamped to ends)."""
        lf = np.clip(np.asarray(load_fraction, dtype=float),
                     self.load[0], self.load[-1])
        return np.interp(lf, self.load, self.eff)

    def losses_fraction(self, load_fraction):
        """Loss as a fraction of input power (1 - eta) * load."""
        eta = self.efficiency(load_fraction)
        lf = np.clip(np.asarray(load_fraction, dtype=float),
                     self.load[0], self.load[-1])
        return lf * (1.0 / eta - 1.0)
