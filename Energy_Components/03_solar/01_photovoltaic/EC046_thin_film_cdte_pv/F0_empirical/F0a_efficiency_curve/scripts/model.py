"""F0a empirical efficiency-curve model for EC046 Thin-Film CdTe PV.

Fidelity F0 (empirical lookup). Pure NumPy.

Physics is reduced to a single faithful empirical relation::

    P = eta_stc * f_irr(G) * (1 + gamma_pmp*(T_cell - 25)) * G * A

where ``f_irr(G)`` is a tabulated relative-efficiency curve capturing the
low-light efficiency fall-off, ``gamma_pmp`` is the datasheet power
temperature coefficient and ``eta_stc = P_mp_stc / (1000 * A)``.

Data source: First Solar Series 6 datasheet.
"""
import numpy as np


class EfficiencyCurve:
    def __init__(self, eta_stc, gamma_pmp, area, irr_bp, rel_eff_bp,
                 bifaciality=0.0, stc_temp=25.0):
        self.eta_stc = float(eta_stc)
        self.gamma_pmp = float(gamma_pmp)
        self.area = float(area)
        self.irr_bp = np.asarray(irr_bp, dtype=float)
        self.rel_eff_bp = np.asarray(rel_eff_bp, dtype=float)
        self.bifaciality = float(bifaciality)
        self.stc_temp = float(stc_temp)

    def rel_eff(self, g):
        """Relative efficiency factor (0..~1) at irradiance g [W/m2]."""
        g = np.asarray(g, dtype=float)
        return np.interp(g, self.irr_bp, self.rel_eff_bp)

    def efficiency(self, g, t_cell=25.0):
        """Module efficiency at irradiance g and cell temperature t_cell."""
        g = np.asarray(g, dtype=float)
        eta = self.eta_stc * self.rel_eff(g) * (1.0 + self.gamma_pmp * (t_cell - self.stc_temp))
        return np.clip(eta, 0.0, 1.0)

    def power(self, g, t_cell=25.0, g_rear=0.0):
        """DC power [W]. g_rear adds bifacial rear contribution."""
        g = np.asarray(g, dtype=float)
        g_eff = g + self.bifaciality * np.asarray(g_rear, dtype=float)
        return self.efficiency(g_eff, t_cell) * g_eff * self.area
