"""
EC074 — Plate Heat Exchanger — F1a Effectiveness-NTU Model

Counter-flow configuration (maximum effectiveness for given NTU):
    NTU = U * A / C_min
    C_r  = C_min / C_max
    epsilon = (1 - exp(-NTU*(1-C_r))) / (1 - C_r*exp(-NTU*(1-C_r)))   for C_r < 1
    epsilon = NTU / (1 + NTU)                                           for C_r = 1
    Q = epsilon * C_min * (T_h_in - T_c_in)
    T_h_out = T_h_in - Q / C_h
    T_c_out = T_c_in + Q / C_c

References:
    Incropera & DeWitt (2006), Fundamentals of Heat and Mass Transfer, 6th ed., ch. 11.
    Shah & Sekulic (2003), Fundamentals of Heat Exchanger Design, Wiley.
"""

import numpy as np


class PlateHeatExchangerF1a:
    """Counter-flow plate heat exchanger — effectiveness-NTU method."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.U = u["U"]["value"]         # W/m2K
        self.A = u["A"]["value"]         # m2
        self.cp_h = u["cp_hot"]["value"] # J/kgK
        self.cp_c = u["cp_cold"]["value"]# J/kgK

    def predict(self, T_h_in, T_c_in, m_dot_hot, m_dot_cold):
        """
        Parameters
        ----------
        T_h_in, T_c_in  : float or array  [degC]
        m_dot_hot, m_dot_cold : float or array [kg/s]

        Returns
        -------
        dict with Q_kw, T_h_out, T_c_out, effectiveness, ntu
        """
        T_h_in  = np.asarray(T_h_in,    dtype=float)
        T_c_in  = np.asarray(T_c_in,    dtype=float)
        m_dot_h = np.asarray(m_dot_hot,  dtype=float)
        m_dot_c = np.asarray(m_dot_cold, dtype=float)

        C_h   = m_dot_h * self.cp_h   # W/K
        C_c   = m_dot_c * self.cp_c   # W/K
        C_min = np.minimum(C_h, C_c)
        C_max = np.maximum(C_h, C_c)

        # Guard: if either flow rate is zero (or near-zero), there is no heat
        # exchange.  Return Q=0 and outlet temps equal to inlet temps.
        zero_flow = C_min < 1e-10

        # Use safe denominators to avoid division-by-zero in the main
        # calculation path; results for zero-flow entries are overwritten below.
        C_min_safe = np.where(zero_flow, 1.0, C_min)
        C_max_safe = np.where(zero_flow, 1.0, C_max)
        C_h_safe   = np.where(C_h < 1e-10, 1.0, C_h)
        C_c_safe   = np.where(C_c < 1e-10, 1.0, C_c)

        C_r   = C_min_safe / C_max_safe   # capacity ratio

        NTU = self.U * self.A / C_min_safe  # number of transfer units

        # Counter-flow effectiveness — handle C_r == 1 (equal capacity rates)
        # Use a safe C_r (slightly perturbed) for the C_r<1 branch to avoid 0/0
        C_r_safe = np.where(np.abs(C_r - 1.0) < 1e-6, C_r + 1e-8, C_r)
        exp_term  = np.exp(-NTU * (1.0 - C_r_safe))
        eps_Cr_lt1 = (1.0 - exp_term) / (1.0 - C_r_safe * exp_term)
        eps_Cr_eq1 = NTU / (1.0 + NTU)
        epsilon = np.where(np.abs(C_r - 1.0) < 1e-6, eps_Cr_eq1, eps_Cr_lt1)
        epsilon = np.clip(epsilon, 0.0, 1.0)

        dT_max = T_h_in - T_c_in
        Q_W = epsilon * C_min * dT_max      # W
        Q_W = np.maximum(Q_W, 0.0)         # no negative heat transfer

        T_h_out = T_h_in  - Q_W / C_h_safe
        T_c_out = T_c_in  + Q_W / C_c_safe

        # Zero-flow override: no heat exchange, outlets = inlets
        Q_W     = np.where(zero_flow, 0.0, Q_W)
        T_h_out = np.where(zero_flow, T_h_in, T_h_out)
        T_c_out = np.where(zero_flow, T_c_in, T_c_out)
        epsilon = np.where(zero_flow, 0.0, epsilon)
        NTU     = np.where(zero_flow, 0.0, NTU)

        return {
            "Q_kw":          Q_W / 1000.0,
            "T_h_out":       T_h_out,
            "T_c_out":       T_c_out,
            "effectiveness": epsilon,
            "ntu":           NTU,
        }
