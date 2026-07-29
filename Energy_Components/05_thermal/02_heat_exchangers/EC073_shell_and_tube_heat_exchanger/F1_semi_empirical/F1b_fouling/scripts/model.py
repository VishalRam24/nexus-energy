"""
EC073 — Shell-and-Tube Heat Exchanger — F1b Fouling Model

Extends F1a (e-NTU with 1-2 pass correction) with fouling resistance:

    1/UA_eff = 1/UA_0 + R_f_total

where R_f_total = Rf_shell + Rf_tube (both referred to external tube area).

The fouled UA reduces NTU, which reduces the counter-flow 1-2 pass
effectiveness, which reduces Q and both outlet temperatures.

Cleanliness factor:
    CF = UA_eff / UA_0 = 1 / (1 + UA_0 * R_f_total)

TEMA standard fouling resistances (R_GP-T-2.4):
    Treated cooling tower water:     Rf = 0.0001 m2K/W (tube side)
                                         0.0002 m2K/W (shell side)
    River water:                     Rf = 0.0003–0.001 m2K/W
    Seawater (< 50 C):               Rf = 0.0001 m2K/W
    Dirty industrial / steam (raw):  Rf = 0.001–0.005 m2K/W

References:
    Incropera & DeWitt (2006). Fundamentals of Heat and Mass Transfer, ch.11.
    Shah & Sekulic (2003). Fundamentals of Heat Exchanger Design, Wiley, ch.9.
    TEMA Standards, 10th ed. (2007). Table RGP-T-2.4.
    Kern, D.Q. (1950). Process Heat Transfer. McGraw-Hill.
"""

import numpy as np


class ShellTubeHXF1b:
    """Shell-and-tube HX (1 shell / 2 tube passes) with fouling correction."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.U_clean = u["U_clean"]["value"]              # W/m2K
        self.A       = u["A"]["value"]                    # m2
        self.cp_h    = u["cp_hot"]["value"]               # J/kgK
        self.cp_c    = u["cp_cold"]["value"]              # J/kgK
        self.Rf_shell_default = u["Rf_shell_default"]["value"]
        self.Rf_tube_default  = u["Rf_tube_default"]["value"]

    # ------------------------------------------------------------------
    # Effective UA with fouling
    # ------------------------------------------------------------------

    def UA_effective(self, Rf_shell=None, Rf_tube=None):
        """
        UA_eff = 1 / (1/UA_0 + R_f_total)

        R_f_total = Rf_shell + Rf_tube  [m2K/W, both referred to same area A]
        """
        if Rf_shell is None:
            Rf_shell = self.Rf_shell_default
        if Rf_tube is None:
            Rf_tube = self.Rf_tube_default
        Rf_shell = np.asarray(Rf_shell, dtype=float)
        Rf_tube  = np.asarray(Rf_tube,  dtype=float)
        R_f_total = Rf_shell + Rf_tube
        # 1/UA_0 = 1/(U_clean * A)
        UA_0 = self.U_clean * self.A
        return UA_0 / (1.0 + UA_0 * R_f_total)

    def cleanliness_factor(self, Rf_shell=None, Rf_tube=None):
        """CF = UA_eff / UA_0 (1.0 = clean, < 1.0 = fouled)."""
        UA_eff = self.UA_effective(Rf_shell, Rf_tube)
        return UA_eff / (self.U_clean * self.A)

    # ------------------------------------------------------------------
    # 1-2 pass shell-and-tube effectiveness (e-NTU)
    # ------------------------------------------------------------------

    @staticmethod
    def _effectiveness_1_2(NTU, C_r):
        """
        Counter-flow 1-shell-pass, 2-tube-pass effectiveness (Incropera eq.11.28):

            eps = 2 / { 1 + C_r + sqrt(1+C_r^2) * [1+exp(-NTU*sqrt(1+C_r^2))]
                                                    / [1-exp(-NTU*sqrt(1+C_r^2))] }
        """
        NTU = np.asarray(NTU, dtype=float)
        C_r = np.asarray(C_r, dtype=float)
        # Clamp C_r away from exact 0 to avoid /0
        C_r_safe = np.maximum(C_r, 1e-10)
        sq = np.sqrt(1.0 + C_r_safe ** 2)
        exp_term = np.exp(-NTU * sq)
        # Guard exp_term from being exactly 1 (large NTU, small sq)
        denom_safe = np.where(np.abs(1.0 - exp_term) < 1e-12, 1e-12, 1.0 - exp_term)
        frac = (1.0 + exp_term) / denom_safe
        eps = 2.0 / (1.0 + C_r_safe + sq * frac)
        return np.clip(eps, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Main predict
    # ------------------------------------------------------------------

    def predict(self, T_h_in, T_c_in, m_dot_hot, m_dot_cold,
                Rf_shell=None, Rf_tube=None):
        """
        Parameters
        ----------
        T_h_in, T_c_in   : float or array [degC]
        m_dot_hot/cold    : float or array [kg/s]
        Rf_shell, Rf_tube : float or array [m2K/W]

        Returns
        -------
        dict : Q_kw, T_h_out, T_c_out, effectiveness, ntu, UA_effective,
               cleanliness_factor, effectiveness_reduction
        """
        T_h_in  = np.asarray(T_h_in,   dtype=float)
        T_c_in  = np.asarray(T_c_in,   dtype=float)
        m_dot_h = np.asarray(m_dot_hot, dtype=float)
        m_dot_c = np.asarray(m_dot_cold, dtype=float)

        UA_eff   = self.UA_effective(Rf_shell, Rf_tube)
        UA_clean = self.U_clean * self.A

        C_h = m_dot_h * self.cp_h
        C_c = m_dot_c * self.cp_c
        C_min = np.minimum(C_h, C_c)
        C_max = np.maximum(C_h, C_c)

        zero_flow = C_min < 1e-10
        C_min_s = np.where(zero_flow, 1.0, C_min)
        C_max_s = np.where(zero_flow, 1.0, C_max)
        C_h_s   = np.where(C_h < 1e-10, 1.0, C_h)
        C_c_s   = np.where(C_c < 1e-10, 1.0, C_c)

        C_r = C_min_s / C_max_s

        NTU_fouled = UA_eff  * self.A / C_min_s  # Note: UA_eff already has A baked in
        NTU_clean  = UA_clean / C_min_s
        # Correct: NTU = UA_eff / C_min (UA_eff already has units W/K)
        NTU_fouled = UA_eff  / C_min_s
        NTU_clean  = UA_clean / C_min_s

        eps_fouled = self._effectiveness_1_2(NTU_fouled, C_r)
        eps_clean  = self._effectiveness_1_2(NTU_clean,  C_r)

        dT_max = T_h_in - T_c_in
        Q_W = eps_fouled * C_min * dT_max
        Q_W = np.maximum(Q_W, 0.0)

        T_h_out = T_h_in - Q_W / C_h_s
        T_c_out = T_c_in + Q_W / C_c_s

        eps_reduction = np.where(
            eps_clean > 1e-10,
            (eps_clean - eps_fouled) / eps_clean,
            0.0,
        )

        # Override zero-flow points
        Q_W        = np.where(zero_flow, 0.0, Q_W)
        T_h_out    = np.where(zero_flow, T_h_in, T_h_out)
        T_c_out    = np.where(zero_flow, T_c_in, T_c_out)
        eps_fouled = np.where(zero_flow, 0.0, eps_fouled)
        NTU_fouled = np.where(zero_flow, 0.0, NTU_fouled)

        return {
            "Q_kw":               Q_W / 1000.0,
            "T_h_out":            T_h_out,
            "T_c_out":            T_c_out,
            "effectiveness":      eps_fouled,
            "ntu":                NTU_fouled,
            "UA_effective":       UA_eff,
            "cleanliness_factor": self.cleanliness_factor(Rf_shell, Rf_tube),
            "effectiveness_reduction": eps_reduction,
        }
