"""
EC073 — Shell-and-Tube Heat Exchanger — F1a LMTD/NTU Model

1 shell pass / 2 tube passes (TEMA E) — the most common shell-and-tube
configuration. Uses the closed-form effectiveness-NTU expression for the
1-2 layout (Bowman, Mueller & Nagle, 1940; Incropera Eq. 11.30b):

    NTU  = U * A / C_min
    C_r  = C_min / C_max

    epsilon_1-2 = 2 * { (1 + C_r) + sqrt(1 + C_r**2)
                    * (1 + exp(-NTU*sqrt(1+C_r**2)))
                    / (1 - exp(-NTU*sqrt(1+C_r**2))) }^{-1}

Heat duty:

    Q = epsilon * C_min * (T_h_in - T_c_in)

Outlet temperatures from species energy balances. The LMTD correction
factor F (1 shell, 2 tube passes) is also reported as a diagnostic; the
core heat-duty calculation uses ε-NTU which is numerically robust.

References:
    Incropera & DeWitt (2006), ch.11.
    Bowman, R.A., Mueller, A.C., Nagle, W.M. (1940). Trans. ASME 62, 283-294.
    Kakac, S., Liu, H. (2002). Heat Exchangers, 2nd ed., CRC Press.
"""

import numpy as np


class ShellAndTubeHEX1a:
    """Shell-and-tube (1 shell, 2 tube passes) heat exchanger — TEMA E."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.U = u["U"]["value"]            # W/m2K
        self.A = u["A"]["value"]            # m2
        self.cp_h = u["cp_hot"]["value"]
        self.cp_c = u["cp_cold"]["value"]
        self.n_shell = int(u["n_shell_passes"]["value"])
        self.n_tube  = int(u["n_tube_passes"]["value"])

    # ------------------------------------------------------------------
    @staticmethod
    def _effectiveness_1_2(NTU, C_r):
        """1-shell, 2-tube-pass effectiveness."""
        s = np.sqrt(1.0 + C_r * C_r)
        # Guard against extreme NTU producing overflow in exp
        arg = NTU * s
        # Use stable formulation: (1+e^-x)/(1-e^-x) = coth(x/2)
        ex = np.exp(-arg)
        ratio = (1.0 + ex) / np.where(1.0 - ex < 1e-12, 1e-12, 1.0 - ex)
        denom = (1.0 + C_r) + s * ratio
        return 2.0 / denom

    @staticmethod
    def _F_correction_1_2(P, R):
        """LMTD correction factor for 1 shell pass, 2 tube passes (TEMA E).

        P = (T_c_out - T_c_in) / (T_h_in - T_c_in)
        R = (T_h_in - T_h_out) / (T_c_out - T_c_in)
        """
        # Avoid singularities
        P = np.clip(P, 1e-9, 1.0 - 1e-9)
        R_safe = np.where(np.abs(R - 1.0) < 1e-6, 1.0 + 1e-6, R)
        s = np.sqrt(R_safe ** 2 + 1.0)
        num = s * np.log((1.0 - P) / (1.0 - P * R_safe))
        denom_log_arg_num = 2.0 - P * (R_safe + 1.0 - s)
        denom_log_arg_den = 2.0 - P * (R_safe + 1.0 + s)
        denom_log_arg = np.where(denom_log_arg_den > 1e-12,
                                 denom_log_arg_num / denom_log_arg_den, 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            denom = (R_safe - 1.0) * np.log(np.where(denom_log_arg > 1e-12, denom_log_arg, 1e-12))
            F = np.where(np.abs(denom) > 1e-12, num / np.where(np.abs(denom) > 1e-12, denom, 1.0), 1.0)
        F = np.where(np.isfinite(F), F, 1.0)
        return np.clip(F, 0.0, 1.0)

    # ------------------------------------------------------------------
    def predict(self, T_h_in, T_c_in, m_dot_hot, m_dot_cold):
        T_h_in = np.asarray(T_h_in, dtype=float)
        T_c_in = np.asarray(T_c_in, dtype=float)
        m_h    = np.asarray(m_dot_hot,  dtype=float)
        m_c    = np.asarray(m_dot_cold, dtype=float)

        C_h = m_h * self.cp_h
        C_c = m_c * self.cp_c
        C_min = np.minimum(C_h, C_c)
        C_max = np.maximum(C_h, C_c)

        zero_flow = C_min < 1e-10
        C_min_safe = np.where(zero_flow, 1.0, C_min)
        C_max_safe = np.where(zero_flow, 1.0, C_max)
        C_h_safe   = np.where(C_h < 1e-10, 1.0, C_h)
        C_c_safe   = np.where(C_c < 1e-10, 1.0, C_c)

        C_r = C_min_safe / C_max_safe
        NTU = self.U * self.A / C_min_safe
        epsilon = self._effectiveness_1_2(NTU, C_r)
        epsilon = np.clip(epsilon, 0.0, 1.0)

        dT_max = T_h_in - T_c_in
        Q_W = epsilon * C_min * dT_max
        Q_W = np.maximum(Q_W, 0.0)

        T_h_out = T_h_in - Q_W / C_h_safe
        T_c_out = T_c_in + Q_W / C_c_safe

        # LMTD diagnostics (counter-flow LMTD, then F correction)
        dT1 = T_h_in - T_c_out
        dT2 = T_h_out - T_c_in
        dT1_s = np.where(np.abs(dT1) < 1e-9, 1e-9, dT1)
        dT2_s = np.where(np.abs(dT2) < 1e-9, 1e-9, dT2)
        ratio_arg = np.where(dT1_s / dT2_s > 1e-12, dT1_s / dT2_s, 1e-12)
        with np.errstate(invalid="ignore", divide="ignore"):
            log_ratio = np.log(ratio_arg)
        LMTD_cf = np.where(np.abs(dT1 - dT2) > 1e-6,
                            (dT1 - dT2) / np.where(np.abs(log_ratio) > 1e-12, log_ratio, 1e-12),
                            0.5 * (dT1 + dT2))

        # F correction factor for 1-2 layout
        denom_P = T_h_in - T_c_in
        P = np.where(np.abs(denom_P) > 1e-9, (T_c_out - T_c_in) / np.where(np.abs(denom_P) > 1e-9, denom_P, 1.0), 0.0)
        denom_R = (T_c_out - T_c_in)
        R = np.where(np.abs(denom_R) > 1e-9, (T_h_in - T_h_out) / np.where(np.abs(denom_R) > 1e-9, denom_R, 1.0), 1.0)
        F = self._F_correction_1_2(P, R)
        LMTD = F * LMTD_cf

        # Zero-flow override
        Q_W     = np.where(zero_flow, 0.0, Q_W)
        T_h_out = np.where(zero_flow, T_h_in, T_h_out)
        T_c_out = np.where(zero_flow, T_c_in, T_c_out)
        epsilon = np.where(zero_flow, 0.0, epsilon)
        NTU     = np.where(zero_flow, 0.0, NTU)
        LMTD    = np.where(zero_flow, 0.0, LMTD)
        F       = np.where(zero_flow, 1.0, F)

        return {
            "Q_kw":          Q_W / 1000.0,
            "T_h_out":       T_h_out,
            "T_c_out":       T_c_out,
            "effectiveness": epsilon,
            "ntu":           NTU,
            "lmtd":          LMTD,
            "f_correction":  F,
        }
