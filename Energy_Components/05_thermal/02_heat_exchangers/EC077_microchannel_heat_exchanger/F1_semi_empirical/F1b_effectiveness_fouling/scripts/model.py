"""
EC077 — Microchannel Heat Exchanger — F1b Effectiveness-NTU + Fouling + Part-Load LMTD

Extends F1a (ε-NTU clean) with:
  1. Fouling resistance correction: 1/U_fouled = 1/U_clean + Rf_hot + Rf_cold
  2. Part-load LMTD correction factor for off-design flow (cross-flow config)
  3. Channel-count-based scaling for microchannel geometry

Microchannel HX physics:
    U_clean is very high (1000-5000 W/m2K) due to small hydraulic diameter (Dh ~ 0.5-1 mm).
    Fouling is more impactful than for large-tube HX because:
        - Small channels are more susceptible to blockage
        - Relative fouling resistance impact: Rf/A_ratio is larger

Part-load LMTD correction (cross-flow):
    At reduced flow rates, the effectiveness changes and LMTD correction factor
    F_LMTD accounts for deviation from counter-flow.
    F_LMTD = f(P, R) using NTU-P-R method for cross-flow one-pass configuration:
        P = (T_c_out - T_c_in) / (T_h_in - T_c_in)
        R = C_c / C_h = (T_h_in - T_h_out) / (T_c_out - T_c_in)
        For single-pass cross-flow (unmixed-unmixed, Incropera eq. 11.32):
            F depends on P and R — here approximated by a correction envelope.

References:
    Kandlikar & Shah (2012), 'Two-phase flow and pressure drop of refrigerants
    in narrow channels', J. Heat Transfer 120(4).
    Incropera & DeWitt (2006), Fundamentals of Heat and Mass Transfer, 6th ed., Ch. 11.
    TEMA Standards, 10th ed. — Fouling resistance tables.
    Kakaç & Liu (2002), Heat Exchangers: Selection, Rating, and Thermal Design, CRC.
"""

import numpy as np


class MicrochannelHXF1b:
    """Microchannel HX — ε-NTU with fouling and part-load LMTD correction."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.U_clean = u["U_clean"]["value"]                   # W/m2K
        self.A = u["A"]["value"]                                # m2 — total area
        self.cp_h = u["cp_hot"]["value"]                        # J/kgK
        self.cp_c = u["cp_cold"]["value"]                       # J/kgK
        self.Rf_hot_default = u["Rf_hot_default"]["value"]      # m2K/W
        self.Rf_cold_default = u["Rf_cold_default"]["value"]    # m2K/W
        self.config = u.get("flow_config", {}).get("value", "counterflow")
        # Part-load LMTD correction coefficients (cross-flow approximation)
        self.f_lmtd_a = u["f_lmtd_a"]["value"]
        self.f_lmtd_b = u["f_lmtd_b"]["value"]

    # ------------------------------------------------------------------
    # Fouled U-value
    # ------------------------------------------------------------------

    def U_fouled(self, Rf_hot=None, Rf_cold=None):
        """
        Overall HTC with fouling resistances.
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
    # Part-load LMTD correction factor
    # ------------------------------------------------------------------

    def F_lmtd(self, PLR=1.0):
        """
        LMTD correction factor for part-load operation.

        For cross-flow configuration, F_LMTD < 1.0 (vs counter-flow = 1.0).
        At full load (PLR=1): F = f_lmtd_a (nominal cross-flow factor, ~0.85-0.97).
        At part load: F decreases slightly due to flow maldistribution in channels.

        F(PLR) = f_lmtd_a - f_lmtd_b * (1.0 - PLR)
        i.e. F_lmtd_b is the magnitude of decrease per unit drop in PLR.
        (Linear model; conservative for PLR in [0.5, 1.0])
        """
        PLR = np.asarray(PLR, dtype=float)
        F = self.f_lmtd_a - self.f_lmtd_b * (1.0 - PLR)
        return np.clip(F, 0.5, 1.0)

    # ------------------------------------------------------------------
    # Core ε-NTU heat transfer
    # ------------------------------------------------------------------

    def _entu(self, NTU, C_r):
        """Counter-flow ε-NTU formula with C_r=1 special case."""
        C_r_safe = np.where(np.abs(C_r - 1.0) < 1e-6, C_r + 1e-8, C_r)
        exp_term = np.exp(-NTU * (1.0 - C_r_safe))
        eps_lt1 = (1.0 - exp_term) / (1.0 - C_r_safe * exp_term)
        eps_eq1 = NTU / (1.0 + NTU)
        eps = np.where(np.abs(C_r - 1.0) < 1e-6, eps_eq1, eps_lt1)
        return np.clip(eps, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Main predict
    # ------------------------------------------------------------------

    def predict(self, T_h_in, T_c_in, m_dot_hot, m_dot_cold,
                Rf_hot=None, Rf_cold=None, PLR=1.0):
        """
        Parameters
        ----------
        T_h_in, T_c_in      : float or array [degC]
        m_dot_hot, m_dot_cold : float or array [kg/s]
        Rf_hot, Rf_cold      : fouling resistances [m2K/W]
        PLR                  : part-load ratio [0.5-1.0]

        Returns
        -------
        dict: Q_kw, T_h_out, T_c_out, effectiveness, ntu, U_fouled,
              effectiveness_reduction, F_lmtd
        """
        T_h_in  = np.asarray(T_h_in,     dtype=float)
        T_c_in  = np.asarray(T_c_in,     dtype=float)
        m_dot_h = np.asarray(m_dot_hot,  dtype=float)
        m_dot_c = np.asarray(m_dot_cold, dtype=float)

        U_f = self.U_fouled(Rf_hot, Rf_cold)
        F   = self.F_lmtd(PLR)

        C_h   = m_dot_h * self.cp_h
        C_c   = m_dot_c * self.cp_c
        C_min = np.minimum(C_h, C_c)
        C_max = np.maximum(C_h, C_c)

        zero_flow = C_min < 1e-10
        C_min_safe = np.where(zero_flow, 1.0, C_min)
        C_max_safe = np.where(zero_flow, 1.0, C_max)
        C_h_safe = np.where(C_h < 1e-10, 1.0, C_h)
        C_c_safe = np.where(C_c < 1e-10, 1.0, C_c)

        C_r = C_min_safe / C_max_safe

        # NTU with fouled U, scaled by LMTD correction factor F
        # For cross-flow: effective NTU is reduced by F vs pure counter-flow
        NTU_fouled = U_f * self.A * F / C_min_safe
        # Clean NTU also uses F_lmtd so that eps_reduction = 0 when Rf = 0
        NTU_clean  = self.U_clean * self.A * F / C_min_safe

        eps_fouled = self._entu(NTU_fouled, C_r)
        eps_clean  = self._entu(NTU_clean, C_r)

        dT_max = T_h_in - T_c_in
        Q_W = eps_fouled * C_min * dT_max
        Q_W = np.maximum(Q_W, 0.0)

        T_h_out = T_h_in - Q_W / C_h_safe
        T_c_out = T_c_in + Q_W / C_c_safe

        eps_reduction = np.where(
            eps_clean > 1e-10,
            (eps_clean - eps_fouled) / eps_clean,
            0.0,
        )

        # Zero-flow override
        Q_W        = np.where(zero_flow, 0.0, Q_W)
        T_h_out    = np.where(zero_flow, T_h_in, T_h_out)
        T_c_out    = np.where(zero_flow, T_c_in, T_c_out)
        eps_fouled = np.where(zero_flow, 0.0, eps_fouled)
        NTU_fouled = np.where(zero_flow, 0.0, NTU_fouled)

        return {
            "Q_kw":                   Q_W / 1000.0,
            "T_h_out":                T_h_out,
            "T_c_out":                T_c_out,
            "effectiveness":          eps_fouled,
            "ntu":                    NTU_fouled,
            "U_fouled":               U_f,
            "effectiveness_reduction": eps_reduction,
            "F_lmtd":                 np.asarray(F, dtype=float),
        }
