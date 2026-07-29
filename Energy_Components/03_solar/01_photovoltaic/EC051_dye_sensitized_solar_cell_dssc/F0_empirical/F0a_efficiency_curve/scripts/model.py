"""F0a empirical rated-efficiency model for EC051 Dye-Sensitized Solar Cell (DSSC).

Fidelity F0 (empirical lookup). Pure NumPy.

    eta(G,T) = eta_stc * f_irr(G) * (1 + gamma*(T - 25))
    P = eta(G,T) * G   (per unit cell area; multiply by area externally)

``f_irr`` is a tabulated relative-efficiency curve (low-light fall-off).

Data source: O'Regan & Graetzel (1991) Nature; Graetzel (2003).
"""
import numpy as np


class EfficiencyCurve:
    def __init__(self, eta_stc, gamma, irr_bp, rel_eff_bp, stc_temp=25.0):
        self.eta_stc = float(eta_stc)
        self.gamma = float(gamma)
        self.irr_bp = np.asarray(irr_bp, dtype=float)
        self.rel_eff_bp = np.asarray(rel_eff_bp, dtype=float)
        self.stc_temp = float(stc_temp)

    def rel_eff(self, g):
        g = np.asarray(g, dtype=float)
        return np.interp(g, self.irr_bp, self.rel_eff_bp)

    def efficiency(self, g, t_cell=25.0):
        g = np.asarray(g, dtype=float)
        eta = self.eta_stc * self.rel_eff(g) * (1.0 + self.gamma * (t_cell - self.stc_temp))
        return np.clip(eta, 0.0, 1.0)

    def power_density(self, g, t_cell=25.0):
        """Power per unit area [W/m2]."""
        g = np.asarray(g, dtype=float)
        return self.efficiency(g, t_cell) * g
