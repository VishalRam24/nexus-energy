"""
EC075 — Finned-Tube Heat Exchanger — F1a Effectiveness-NTU

Simplest e-NTU model for cross-flow finned-tube HX (one fluid unmixed):
    eps = 1 - exp((NTU^0.22 / C_r) * (exp(-C_r * NTU^0.78) - 1))
    Q = eps * C_min * (T_h_in - T_c_in)

No fouling, no property corrections — pure geometry + constant U.

References:
    Incropera & DeWitt (2006). Fundamentals of Heat and Mass Transfer, ch.11.
    Kays & London (1984). Compact Heat Exchanger Design.
"""

import numpy as np


class FinnedTubeHXF1a:
    """Cross-flow finned-tube HX using constant-U e-NTU method."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.U = u["U_overall"]["value"]       # W/m2K
        self.A = u["A"]["value"]               # m2 total external area
        self.cp_h = u["cp_hot"]["value"]       # J/kgK
        self.cp_c = u["cp_cold"]["value"]      # J/kgK

    @staticmethod
    def _effectiveness_crossflow(NTU, C_r):
        """Cross-flow, one fluid unmixed (Incropera eq. 11.32)."""
        NTU = np.asarray(NTU, dtype=float)
        C_r = np.asarray(C_r, dtype=float)
        C_r_safe = np.maximum(C_r, 1e-10)
        inner = np.exp(-C_r_safe * NTU ** 0.78) - 1.0
        eps = 1.0 - np.exp((NTU ** 0.22 / C_r_safe) * inner)
        return np.clip(eps, 0.0, 1.0)

    def predict(self, T_h_in, T_c_in, m_dot_hot, m_dot_cold):
        T_h_in = np.asarray(T_h_in, dtype=float)
        T_c_in = np.asarray(T_c_in, dtype=float)
        m_dot_h = np.asarray(m_dot_hot, dtype=float)
        m_dot_c = np.asarray(m_dot_cold, dtype=float)

        C_h = m_dot_h * self.cp_h
        C_c = m_dot_c * self.cp_c
        C_min = np.minimum(C_h, C_c)
        C_max = np.maximum(C_h, C_c)

        zero = C_min < 1e-10
        C_min_s = np.where(zero, 1.0, C_min)
        C_max_s = np.where(zero, 1.0, C_max)
        C_r = C_min_s / C_max_s

        NTU = self.U * self.A / C_min_s
        eps = self._effectiveness_crossflow(NTU, C_r)

        Q_W = eps * C_min * (T_h_in - T_c_in)
        Q_W = np.maximum(Q_W, 0.0)

        C_h_s = np.where(C_h < 1e-10, 1.0, C_h)
        C_c_s = np.where(C_c < 1e-10, 1.0, C_c)
        T_h_out = T_h_in - Q_W / C_h_s
        T_c_out = T_c_in + Q_W / C_c_s

        Q_W = np.where(zero, 0.0, Q_W)
        T_h_out = np.where(zero, T_h_in, T_h_out)
        T_c_out = np.where(zero, T_c_in, T_c_out)
        eps = np.where(zero, 0.0, eps)

        return {
            "Q_kw": Q_W / 1000.0,
            "T_h_out": T_h_out,
            "T_c_out": T_c_out,
            "effectiveness": eps,
            "NTU": NTU,
        }
