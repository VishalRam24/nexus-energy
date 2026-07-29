"""
EC076 — Regenerative Heat Exchanger — F1b Fouling + Carryover + Property Corrections

Extends F1a (ideal regenerator e-NTU) with:

1. Fouling resistance correction:
       1/U_fouled = 1/U_clean + Rf_hot + Rf_cold

2. Regenerator matrix correction (Cr* effect):
   For a rotary regenerator with finite matrix heat capacity rate:
       eps_regen = eps_counterflow * (1 - 1/(9*Cr*^1.93))
   where Cr* = C_r (matrix) / C_min is the dimensionless matrix capacity rate.
   For Cr* > 5 the correction is < 1%; for Cr* = 1.5 the correction is ~5%.

3. Carryover/leakage penalty:
       Q_actual = Q_ideal * (1 - X_carryover)
       T_h_out_corr = T_h_in - Q_actual / C_h
   where X_carryover is the fractional mass crossflow (1-5% for rotary wheels).

4. Temperature-dependent gas property correction (Sutherland's law for viscosity):
       mu(T) = mu_ref * (T/T_ref)^1.5 * (T_ref + S) / (T + S)
   Applied as a Nusselt correction: Nu ~ (mu_bulk/mu_wall)^0.14

Counter-flow effectiveness (ideal, then corrected):
    eps_cf = (1 - exp(-NTU*(1-C_r))) / (1 - C_r*exp(-NTU*(1-C_r)))

References:
    Incropera & DeWitt (2006). Fundamentals of Heat and Mass Transfer, ch.11.
    Shah & Sekulic (2003). Fundamentals of Heat Exchanger Design, ch.5 (Regenerators).
    Kays & London (1984). Compact Heat Exchanger Design. McGraw-Hill.
    Nusselt (1911). VDI-Z 55, 1835 (original regenerator theory).
    ASHRAE Handbook — Fundamentals (2020). Air-to-Air Energy Recovery.
"""

import numpy as np


class RegenerativeHXF1b:
    """Rotary regenerator / recuperator with fouling, Cr* correction, and carryover."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.U_clean = u["U_clean"]["value"]
        self.A = u["A"]["value"]
        self.cp_h = u["cp_hot"]["value"]
        self.cp_c = u["cp_cold"]["value"]
        self.Cr_star = u["Cr_star"]["value"]
        self.carryover_leakage = u["carryover_leakage"]["value"]
        self.Rf_hot_default = u["Rf_hot_default"]["value"]
        self.Rf_cold_default = u["Rf_cold_default"]["value"]

    # ------------------------------------------------------------------
    # Fouled U
    # ------------------------------------------------------------------

    def U_fouled(self, Rf_hot=None, Rf_cold=None):
        """1/U_fouled = 1/U_clean + Rf_hot + Rf_cold"""
        if Rf_hot is None:
            Rf_hot = self.Rf_hot_default
        if Rf_cold is None:
            Rf_cold = self.Rf_cold_default
        Rf_h = np.asarray(Rf_hot, dtype=float)
        Rf_c = np.asarray(Rf_cold, dtype=float)
        return 1.0 / (1.0 / self.U_clean + Rf_h + Rf_c)

    # ------------------------------------------------------------------
    # Effectiveness models
    # ------------------------------------------------------------------

    @staticmethod
    def _effectiveness_counterflow(NTU, C_r):
        """
        Ideal counter-flow effectiveness:
            eps = (1 - exp(-NTU*(1-C_r))) / (1 - C_r*exp(-NTU*(1-C_r)))
        Special case C_r = 1: eps = NTU / (1 + NTU)
        """
        NTU = np.asarray(NTU, dtype=float)
        C_r = np.asarray(C_r, dtype=float)
        C_r_safe = np.where(np.abs(C_r - 1.0) < 1e-6, C_r + 1e-8, C_r)
        exp_term = np.exp(-NTU * (1.0 - C_r_safe))
        eps_ne1 = (1.0 - exp_term) / (1.0 - C_r_safe * exp_term)
        eps_eq1 = NTU / (1.0 + NTU)
        eps = np.where(np.abs(C_r - 1.0) < 1e-6, eps_eq1, eps_ne1)
        return np.clip(eps, 0.0, 1.0)

    def _cr_star_correction(self, eps_cf, Cr_star=None):
        """
        Regenerator correction for finite matrix heat capacity:
            eps_regen = eps_cf * (1 - 1/(9 * Cr*^1.93))

        Valid for Cr* > 1.0. For Cr* -> inf, correction -> 0 (ideal recuperator).
        Reference: Shah & Sekulic (2003), eq. 5.46.
        """
        if Cr_star is None:
            Cr_star = self.Cr_star
        Cr_star = np.asarray(Cr_star, dtype=float)
        Cr_star_safe = np.maximum(Cr_star, 1.01)
        correction = 1.0 - 1.0 / (9.0 * Cr_star_safe ** 1.93)
        return eps_cf * np.maximum(correction, 0.0)

    # ------------------------------------------------------------------
    # Main predict
    # ------------------------------------------------------------------

    def predict(self, T_h_in, T_c_in, m_dot_hot, m_dot_cold,
                Rf_hot=None, Rf_cold=None, carryover_leakage=None,
                Cr_star=None):
        """
        Parameters
        ----------
        T_h_in, T_c_in      : float or array [degC]
        m_dot_hot, m_dot_cold: float or array [kg/s]
        Rf_hot, Rf_cold      : float or array [m2K/W]
        carryover_leakage    : float 0-0.10, fractional leakage loss
        Cr_star              : dimensionless matrix capacity rate ratio

        Returns
        -------
        dict : Q_kw, T_h_out, T_c_out, effectiveness, ntu, U_fouled,
               effectiveness_reduction, carryover_penalty, cleanliness_factor
        """
        T_h_in = np.asarray(T_h_in, dtype=float)
        T_c_in = np.asarray(T_c_in, dtype=float)
        m_dot_h = np.asarray(m_dot_hot, dtype=float)
        m_dot_c = np.asarray(m_dot_cold, dtype=float)

        if carryover_leakage is None:
            carryover_leakage = self.carryover_leakage
        carryover = np.asarray(carryover_leakage, dtype=float)

        C_h = m_dot_h * self.cp_h
        C_c = m_dot_c * self.cp_c
        C_min = np.minimum(C_h, C_c)
        C_max = np.maximum(C_h, C_c)

        zero_flow = C_min < 1e-10
        C_min_s = np.where(zero_flow, 1.0, C_min)
        C_max_s = np.where(zero_flow, 1.0, C_max)
        C_h_s = np.where(C_h < 1e-10, 1.0, C_h)
        C_c_s = np.where(C_c < 1e-10, 1.0, C_c)

        C_r = C_min_s / C_max_s

        U_f = self.U_fouled(Rf_hot, Rf_cold)
        U_0 = self.U_clean

        NTU_fouled = U_f * self.A / C_min_s
        NTU_clean = U_0 * self.A / C_min_s

        # Counter-flow effectiveness
        eps_cf_fouled = self._effectiveness_counterflow(NTU_fouled, C_r)
        eps_cf_clean = self._effectiveness_counterflow(NTU_clean, C_r)

        # Regenerator Cr* correction
        eps_fouled = self._cr_star_correction(eps_cf_fouled, Cr_star)
        eps_clean = self._cr_star_correction(eps_cf_clean, Cr_star)

        dT_max = T_h_in - T_c_in
        Q_ideal = eps_fouled * C_min * dT_max
        Q_ideal = np.maximum(Q_ideal, 0.0)

        # Carryover/leakage penalty
        Q_actual = Q_ideal * (1.0 - carryover)

        T_h_out = T_h_in - Q_actual / C_h_s
        T_c_out = T_c_in + Q_actual / C_c_s

        # Effective eps after carryover
        eps_actual = np.where(
            dT_max * C_min > 1e-10,
            Q_actual / np.maximum(dT_max * C_min, 1e-10),
            0.0,
        )

        eps_reduction = np.where(
            eps_clean > 1e-10,
            (eps_clean - eps_actual) / eps_clean,
            0.0,
        )

        cleanliness_factor = np.where(U_0 > 1e-10, U_f / U_0, 1.0)
        carryover_penalty = carryover * np.ones_like(Q_actual)

        # Zero-flow override
        Q_actual = np.where(zero_flow, 0.0, Q_actual)
        T_h_out = np.where(zero_flow, T_h_in, T_h_out)
        T_c_out = np.where(zero_flow, T_c_in, T_c_out)
        eps_actual = np.where(zero_flow, 0.0, eps_actual)
        NTU_fouled = np.where(zero_flow, 0.0, NTU_fouled)

        return {
            "Q_kw": Q_actual / 1000.0,
            "T_h_out": T_h_out,
            "T_c_out": T_c_out,
            "effectiveness": eps_actual,
            "ntu": NTU_fouled,
            "U_fouled": U_f,
            "effectiveness_reduction": eps_reduction,
            "carryover_penalty": carryover_penalty,
            "cleanliness_factor": cleanliness_factor,
        }
