"""
EC074 — Plate Heat Exchanger — F1b Fouling Model

Extends F1a (e-NTU method) with fouling resistance:

    1/U_fouled = 1/U_clean + Rf_hot + Rf_cold

The fouled U value is used in NTU = U_fouled * A / C_min, reducing the
effectiveness and heat transfer rate.

Typical fouling resistances (TEMA Standards):
    Clean water:          0.0001 m2K/W
    Treated cooling water: 0.0002 m2K/W
    City/tap water:       0.0003 m2K/W
    Untreated water:      0.0005 m2K/W
    River water:          0.001  m2K/W
    Dirty industrial:     0.005  m2K/W

References:
    Incropera & DeWitt (2006), Fundamentals of Heat and Mass Transfer, ch. 11.
    Shah & Sekulic (2003), Fundamentals of Heat Exchanger Design, Wiley.
    TEMA Standards, 10th ed. — Fouling resistance tables.
"""

import numpy as np


class PlateHeatExchangerF1b:
    """Counter-flow plate heat exchanger with fouling resistance correction."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.U_clean = u["U_clean"]["value"]           # W/m2K
        self.A = u["A"]["value"]                       # m2
        self.cp_h = u["cp_hot"]["value"]               # J/kgK
        self.cp_c = u["cp_cold"]["value"]              # J/kgK
        self.Rf_hot_default = u["Rf_hot_default"]["value"]    # m2K/W
        self.Rf_cold_default = u["Rf_cold_default"]["value"]  # m2K/W

    # ------------------------------------------------------------------
    # Fouled U-value
    # ------------------------------------------------------------------

    def U_fouled(self, Rf_hot=None, Rf_cold=None):
        """
        Overall heat transfer coefficient with fouling.
        1/U_fouled = 1/U_clean + Rf_hot + Rf_cold
        """
        if Rf_hot is None:
            Rf_hot = self.Rf_hot_default
        if Rf_cold is None:
            Rf_cold = self.Rf_cold_default

        Rf_hot = np.asarray(Rf_hot, dtype=float)
        Rf_cold = np.asarray(Rf_cold, dtype=float)

        R_total = 1.0 / self.U_clean + Rf_hot + Rf_cold
        return 1.0 / R_total

    # ------------------------------------------------------------------
    # Core e-NTU calculation
    # ------------------------------------------------------------------

    def predict(self, T_h_in, T_c_in, m_dot_hot, m_dot_cold,
                Rf_hot=None, Rf_cold=None):
        """
        Parameters
        ----------
        T_h_in, T_c_in : float or array [degC]
        m_dot_hot, m_dot_cold : float or array [kg/s]
        Rf_hot, Rf_cold : float or array [m2K/W] — fouling resistances

        Returns
        -------
        dict with Q_kw, T_h_out, T_c_out, effectiveness, ntu,
             U_fouled, effectiveness_reduction
        """
        T_h_in = np.asarray(T_h_in, dtype=float)
        T_c_in = np.asarray(T_c_in, dtype=float)
        m_dot_h = np.asarray(m_dot_hot, dtype=float)
        m_dot_c = np.asarray(m_dot_cold, dtype=float)

        U_f = self.U_fouled(Rf_hot, Rf_cold)

        C_h = m_dot_h * self.cp_h
        C_c = m_dot_c * self.cp_c
        C_min = np.minimum(C_h, C_c)
        C_max = np.maximum(C_h, C_c)

        zero_flow = C_min < 1e-10
        C_min_safe = np.where(zero_flow, 1.0, C_min)
        C_max_safe = np.where(zero_flow, 1.0, C_max)
        C_h_safe = np.where(C_h < 1e-10, 1.0, C_h)
        C_c_safe = np.where(C_c < 1e-10, 1.0, C_c)

        C_r = C_min_safe / C_max_safe

        # NTU with fouled U
        NTU_fouled = U_f * self.A / C_min_safe

        # Also compute clean NTU for effectiveness_reduction
        NTU_clean = self.U_clean * self.A / C_min_safe

        def _effectiveness(NTU, C_r):
            C_r_safe = np.where(np.abs(C_r - 1.0) < 1e-6, C_r + 1e-8, C_r)
            exp_term = np.exp(-NTU * (1.0 - C_r_safe))
            eps_lt1 = (1.0 - exp_term) / (1.0 - C_r_safe * exp_term)
            eps_eq1 = NTU / (1.0 + NTU)
            eps = np.where(np.abs(C_r - 1.0) < 1e-6, eps_eq1, eps_lt1)
            return np.clip(eps, 0.0, 1.0)

        eps_fouled = _effectiveness(NTU_fouled, C_r)
        eps_clean = _effectiveness(NTU_clean, C_r)

        dT_max = T_h_in - T_c_in
        Q_W = eps_fouled * C_min * dT_max
        Q_W = np.maximum(Q_W, 0.0)

        T_h_out = T_h_in - Q_W / C_h_safe
        T_c_out = T_c_in + Q_W / C_c_safe

        # Effectiveness reduction due to fouling
        eps_reduction = np.where(
            eps_clean > 1e-10,
            (eps_clean - eps_fouled) / eps_clean,
            0.0,
        )

        # Zero-flow override
        Q_W = np.where(zero_flow, 0.0, Q_W)
        T_h_out = np.where(zero_flow, T_h_in, T_h_out)
        T_c_out = np.where(zero_flow, T_c_in, T_c_out)
        eps_fouled = np.where(zero_flow, 0.0, eps_fouled)
        NTU_fouled = np.where(zero_flow, 0.0, NTU_fouled)

        return {
            "Q_kw": Q_W / 1000.0,
            "T_h_out": T_h_out,
            "T_c_out": T_c_out,
            "effectiveness": eps_fouled,
            "ntu": NTU_fouled,
            "U_fouled": U_f,
            "effectiveness_reduction": eps_reduction,
        }
