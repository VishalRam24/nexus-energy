"""
EC077 — Microchannel Heat Exchanger — F1a e-NTU Model

Crossflow (both fluids unmixed) e-NTU:
    eps = 1 - exp((NTU^0.22 / C_r) * (exp(-C_r * NTU^0.78) - 1))
    Q = eps * C_min * (T_h_in - T_c_in)

Microchannel HX has very high UA/volume due to small hydraulic diameter (Dh < 1mm).

References:
    Incropera & DeWitt (2006). Fundamentals of Heat and Mass Transfer, eq. 11.32.
    Webb & Kim (2005). Principles of Enhanced Heat Transfer, 2nd ed.
"""

import numpy as np


class MicrochannelHXF1a:
    """Crossflow (both unmixed) e-NTU model for microchannel HX."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.UA = u["UA"]["value"]
        self.C_h_ref = u["C_h"]["value"]
        self.C_c_ref = u["C_c"]["value"]

    @staticmethod
    def _epsilon_crossflow(NTU, C_r):
        """Both fluids unmixed (Incropera eq. 11.32)."""
        NTU = np.asarray(NTU, dtype=float)
        C_r = np.asarray(C_r, dtype=float)
        C_r_safe = np.maximum(C_r, 1e-10)
        inner = np.exp(-C_r_safe * NTU ** 0.78) - 1.0
        eps = 1.0 - np.exp((NTU ** 0.22 / C_r_safe) * inner)
        return np.clip(eps, 0.0, 1.0)

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
        eps = self._epsilon_crossflow(NTU, C_r)

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
