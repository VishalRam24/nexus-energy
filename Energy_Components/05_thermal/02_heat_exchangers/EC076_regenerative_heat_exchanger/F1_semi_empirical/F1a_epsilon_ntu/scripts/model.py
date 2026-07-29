"""
EC076 — Regenerative Heat Exchanger — F1a e-NTU Model

Counterflow e-NTU:
    NTU = UA / C_min
    C_r = C_min / C_max
    eps = (1 - exp(-NTU*(1-C_r))) / (1 - C_r*exp(-NTU*(1-C_r)))   [C_r < 1]
    eps = NTU / (NTU + 1)                                            [C_r = 1]
    Q = eps * C_min * (T_h_in - T_c_in)

References:
    Incropera & DeWitt (2006). Fundamentals of Heat and Mass Transfer, ch.11.
    Kays & London (1984). Compact Heat Exchanger Design.
"""

import numpy as np


class RegenerativeHXF1a:
    """Counterflow e-NTU model for regenerative heat exchanger."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.UA = u["UA"]["value"]
        self.C_h_ref = u["C_h"]["value"]
        self.C_c_ref = u["C_c"]["value"]

    @staticmethod
    def _epsilon_counterflow(NTU, C_r):
        NTU = np.asarray(NTU, dtype=float)
        C_r = np.asarray(C_r, dtype=float)
        balanced = np.abs(C_r - 1.0) < 1e-6
        eps_balanced = NTU / (NTU + 1.0)
        exponent = np.exp(-NTU * (1.0 - C_r))
        eps_unbalanced = (1.0 - exponent) / (1.0 - C_r * exponent + 1e-300)
        return np.where(balanced, eps_balanced, np.clip(eps_unbalanced, 0.0, 1.0))

    def predict(self, T_h_in, T_c_in, C_h=None, C_c=None):
        T_h_in = np.asarray(T_h_in, dtype=float)
        T_c_in = np.asarray(T_c_in, dtype=float)
        C_h = np.asarray(C_h if C_h is not None else self.C_h_ref, dtype=float)
        C_c = np.asarray(C_c if C_c is not None else self.C_c_ref, dtype=float)

        C_min = np.minimum(C_h, C_c)
        C_max = np.maximum(C_h, C_c)
        zero = C_min < 1e-6
        C_min_s = np.where(zero, 1.0, C_min)
        C_max_s = np.where(zero, 1.0, C_max)
        C_r = C_min_s / C_max_s

        NTU = self.UA / C_min_s
        eps = self._epsilon_counterflow(NTU, C_r)

        Q_W = eps * C_min * (T_h_in - T_c_in)
        Q_W = np.maximum(Q_W, 0.0)

        C_h_s = np.where(C_h < 1e-6, 1.0, C_h)
        C_c_s = np.where(C_c < 1e-6, 1.0, C_c)
        T_h_out = T_h_in - Q_W / C_h_s
        T_c_out = T_c_in + Q_W / C_c_s

        Q_W = np.where(zero, 0.0, Q_W)
        eps = np.where(zero, 0.0, eps)
        T_h_out = np.where(zero, T_h_in, T_h_out)
        T_c_out = np.where(zero, T_c_in, T_c_out)

        return {
            "Q_kW": Q_W / 1000.0,
            "T_h_out": T_h_out,
            "T_c_out": T_c_out,
            "effectiveness": eps,
            "NTU": NTU,
        }
