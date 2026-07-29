"""
EC161 -- Dual Active Bridge (DAB) DC-DC Converter -- F1a Ideal Gain + Fixed Efficiency

Ideal voltage gain:
    V_out = N * V_in       (N = N2/N1 turns ratio, unity gain when N=1)

Power transfer via phase-shift modulation (single-phase-shift, SPS):
    P = N * V1 * V2 * phi * (pi - |phi|) / (2 * pi^2 * f * L)
    (positive phi: power flows V1->V2; negative: V2->V1, bidirectional)

With fixed efficiency:
    P_out = eta * P_in  (when V1->V2)
    P_in  = P_out / eta (consumption from V1 side)

References:
    De Doncker, R.W.A.A., Divan, D.M., & Kheraluwala, M.H. (1991).
    IEEE Trans. Ind. Appl., 27(1), 63-73.
"""

import numpy as np


class DABF1a:
    """Dual Active Bridge -- ideal SPS power transfer + fixed efficiency."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.N = u["n_turns"]["value"]
        self.eta = u["eta"]["value"]
        self.f_sw = u["f_sw"]["value"]          # Hz
        self.L = u["L_series"]["value"]         # H
        self.P_rated = u["p_rated"]["value"]    # W
        self.phi_max = u["phi_max"]["value"]    # rad

    def output_voltage(self, v1):
        """V_out = N * V1  (ideal, open-circuit)  [V]."""
        return self.N * np.asarray(v1, dtype=float)

    def power_transfer(self, v1, v2, phi_rad):
        """
        Power from V1 to V2 [W] via single-phase-shift (SPS):
            P = N*V1*V2*phi*(pi-|phi|) / (2*pi^2*f*L)
        Positive = forward, negative = reverse.
        """
        v1 = np.asarray(v1, dtype=float)
        v2 = np.asarray(v2, dtype=float)
        phi = np.clip(np.asarray(phi_rad, dtype=float), -self.phi_max, self.phi_max)
        return (self.N * v1 * v2 * phi * (np.pi - np.abs(phi))) / (2.0 * np.pi**2 * self.f_sw * self.L)

    def efficiency_applied(self, p_transfer):
        """Return (P_out, P_in, P_loss) accounting for direction and efficiency."""
        p = np.asarray(p_transfer, dtype=float)
        p_out = np.where(p >= 0, p * self.eta, p / self.eta)
        p_in = np.where(p >= 0, p, p * self.eta)
        p_loss = np.abs(p_in) - np.abs(p_out)
        return p_out, p_in, p_loss
