"""F0a empirical effectiveness curve for Finned-Tube Heat Exchanger.

Data source: Incropera & DeWitt (2006) ch.11; Kays & London (1984)
Effectiveness eps tabulated vs NTU at the rated capacity ratio Cr.
Heat duty:  Q = eps * Cmin * (T_h_in - T_c_in).  NumPy only.
"""
import numpy as np


class EffectivenessCurve:
    def __init__(self, ntu_bp, eps_bp):
        self.ntu = np.asarray(ntu_bp, dtype=float)
        self.eps = np.asarray(eps_bp, dtype=float)

    def effectiveness(self, ntu):
        n = float(np.clip(ntu, self.ntu[0], self.ntu[-1]))
        return float(np.interp(n, self.ntu, self.eps))
