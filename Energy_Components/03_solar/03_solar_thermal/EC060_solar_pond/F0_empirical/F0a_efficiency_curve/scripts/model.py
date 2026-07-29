"""F0a empirical efficiency-curve model for EC060 Solar Pond.

Fidelity F0 (empirical lookup). Pure NumPy.

Collector/CSP efficiency curve (Hottel-Whillier-Bliss / EN-12975 form)::

    eta(G, dT) = eta0 - a1*dT/G - a2*dT^2/G
    p_out      = eta * G        (useful thermal power per unit aperture area)

with ``dT = T_mean - T_ambient``.  Efficiency is clipped to [0, eta0].

Data source: Tabor (1981) Solar Energy; Hull et al. (1988).
"""
import numpy as np


class EfficiencyCurve:
    def __init__(self, eta0, a1, a2):
        self.eta0 = float(eta0)
        self.a1 = float(a1)
        self.a2 = float(a2)

    def efficiency(self, g, delta_t):
        """Efficiency at irradiance g [W/m2] and dT=T_mean-T_amb [K]."""
        g = np.asarray(g, dtype=float)
        dt = np.asarray(delta_t, dtype=float)
        gg = np.where(g <= 0.0, np.nan, g)
        eta = self.eta0 - self.a1 * dt / gg - self.a2 * dt * dt / gg
        eta = np.nan_to_num(eta, nan=0.0)
        return np.clip(eta, 0.0, self.eta0)

    def power_density(self, g, delta_t):
        """Output power per unit aperture area [W/m2]."""
        g = np.asarray(g, dtype=float)
        return self.efficiency(g, delta_t) * g
