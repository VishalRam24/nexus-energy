"""
EC192 — Gas Pressure Regulator — F1b Choked Flow Model

Phase 7 fix implemented: At choke, the sqrt argument uses ΔP_choke (= Fk*Xt*P_up),
NOT the actual ΔP. This is the correct ISA 75.01.01 formulation.

Physics (ANSI/ISA-75.01.01-2012):
  Critical pressure ratio:
      ΔP_choke = Fk * Xt * P_up
  Choke condition:
      ΔP_actual >= ΔP_choke  ⟹  choked

  Non-choked flow:
      Q = N7 * Cv * Y * sqrt(ΔP_actual * P_up / (G * T * Z))
      Y = 1 - ΔP_actual / (3 * Fk * Xt * P_up)   [≥ 2/3]

  Choked flow (Phase 7 fix):
      Q = N7 * Cv * (2/3) * sqrt(ΔP_choke * P_up / (G * T * Z))
      where ΔP_choke = Fk * Xt * P_up   ← critical ΔP, not actual ΔP

  Key: at choke the flow is independent of downstream pressure; it only depends
  on upstream conditions and the critical ΔP.

Additions over F1a:
  1. Valve travel → effective Cv: Cv_eff = Cv * f(travel)
  2. JT temperature drop (same as F1a) documented here explicitly.
  3. Pressure recovery factor for partial valve opens.

References:
    ANSI/ISA-75.01.01-2012. Flow Equations for Sizing Control Valves.
    Driskell, L.R. (1983). Control Valve Selection and Sizing. ISA.
    Menon, E.S. (2005). Gas Pipeline Hydraulics. CRC Press. Ch. 4.
"""

import numpy as np


class GasPressureRegulatorF1b:
    """
    Gas pressure regulator with correct ISA choked-flow treatment.
    Phase 7 fix: choked flow uses ΔP_choke in sqrt, not actual ΔP.
    """

    # ISA N7 constant: Q [m³/h at std], P [bar], T [K]
    N7 = 4.17e-2

    def __init__(self, params: dict):
        v = params["valve"]
        g = params["gas"]

        self.Cv_default = v["Cv_default"]["value"]
        self.Xt = v["Xt_default"]["value"]
        self.Fk = v["Fk_default"]["value"]
        self.Cv_PLR_coeffs = v["Cv_PLR_coeffs"]["value"]

        self.G = g["specific_gravity"]["value"]
        self.gamma = g["gamma"]["value"]
        self.mu_JT = g["JT_coefficient"]["value"]     # K/Pa
        self.T_inv = g["T_inversion"]["value"]         # K

    # ------------------------------------------------------------------
    # Valve travel → effective Cv
    # ------------------------------------------------------------------
    def cv_effective(self, valve_travel, Cv=None):
        """
        Effective Cv as function of valve travel [0-1].
        Cv_eff = Cv_full * (a0 + a1*travel + a2*travel^2)
        Normalized so travel=1 → Cv_eff = Cv_full.
        """
        travel = np.asarray(valve_travel, dtype=float)
        Cv_full = self.Cv_default if Cv is None else np.asarray(Cv, dtype=float)
        a0, a1, a2 = self.Cv_PLR_coeffs
        f_raw = a0 + a1 * travel + a2 * travel ** 2
        f_at_1 = a0 + a1 + a2
        f = np.clip(f_raw / f_at_1, 0.0, 1.0)
        return Cv_full * f

    # ------------------------------------------------------------------
    # Critical pressure ratio and choke detection
    # ------------------------------------------------------------------
    def dp_choke(self, P_up_bar):
        """Critical pressure drop [bar] at choke: ΔP_choke = Fk * Xt * P_up."""
        return self.Fk * self.Xt * np.asarray(P_up_bar, dtype=float)

    def is_choked(self, P_up_bar, P_down_bar):
        """True where flow is choked."""
        P_up = np.asarray(P_up_bar, dtype=float)
        P_down = np.asarray(P_down_bar, dtype=float)
        return (P_up - P_down) >= self.dp_choke(P_up)

    def expansion_factor_Y(self, P_up_bar, P_down_bar):
        """
        ISA expansion factor Y.
        Non-choked: Y = 1 - ΔP/(3*Fk*Xt*P_up)
        Choked: Y = 2/3 (clamp lower bound)
        """
        P_up = np.asarray(P_up_bar, dtype=float)
        P_down = np.asarray(P_down_bar, dtype=float)
        dP = P_up - P_down
        Y = 1.0 - dP / (3.0 * self.Fk * self.Xt * P_up)
        return np.maximum(Y, 2.0 / 3.0)

    # ------------------------------------------------------------------
    # Flow equation — Phase 7 fix applied
    # ------------------------------------------------------------------
    def flow_std_m3_per_h(self, P_up_bar, P_down_bar, T_up_K, Z=0.9,
                          Cv=None, valve_travel=1.0):
        """
        Gas flow [m³/h std] with correct choked-flow treatment.

        Non-choked:
            Q = N7 * Cv_eff * Y * sqrt(ΔP_actual * P_up / (G * T * Z))

        Choked (Phase 7 fix — uses ΔP_choke, not ΔP_actual):
            Q = N7 * Cv_eff * (2/3) * sqrt(ΔP_choke * P_up / (G * T * Z))

        At choke, flow is independent of P_down. Using ΔP_actual > ΔP_choke in
        sqrt would overpredict flow (non-physical); the correct ISA form locks
        the sqrt argument at the critical value.
        """
        P_up = np.asarray(P_up_bar, dtype=float)
        P_down = np.asarray(P_down_bar, dtype=float)
        T_up = np.asarray(T_up_K, dtype=float)
        Z = np.asarray(Z, dtype=float)
        Cv_eff = self.cv_effective(valve_travel, Cv)

        dP_actual = np.maximum(P_up - P_down, 0.0)
        dP_crit = self.dp_choke(P_up)
        choked = self.is_choked(P_up, P_down)

        # Phase 7 fix: use ΔP_choke (not ΔP_actual) at choke
        dP_eff = np.where(choked, dP_crit, dP_actual)
        Y = self.expansion_factor_Y(P_up, P_down)   # clamped at 2/3 when choked

        Q = self.N7 * Cv_eff * Y * np.sqrt(dP_eff * P_up / (self.G * T_up * Z))
        return Q

    def flow_kg_per_s(self, P_up_bar, P_down_bar, T_up_K, Z=0.9,
                      Cv=None, valve_travel=1.0):
        """Mass flow [kg/s]."""
        Q_m3h = self.flow_std_m3_per_h(P_up_bar, P_down_bar, T_up_K, Z, Cv, valve_travel)
        rho_std = 0.717  # kg/m³ at 288.15 K, 1.01325 bar
        return Q_m3h * rho_std / 3600.0

    def temperature_out(self, T_up_K, P_up_bar, P_down_bar):
        """Downstream temperature after JT expansion [K]."""
        T_up = np.asarray(T_up_K, dtype=float)
        dP_Pa = (np.asarray(P_up_bar, dtype=float) -
                 np.asarray(P_down_bar, dtype=float)) * 1e5
        mu = np.where(T_up < self.T_inv, self.mu_JT, -self.mu_JT)
        return T_up - mu * dP_Pa

    def compute(self, P_up_bar, P_down_bar, T_up_K, Z=0.9,
                Cv=None, valve_travel=1.0):
        """Full computation returning all outputs."""
        Q_m3h = self.flow_std_m3_per_h(P_up_bar, P_down_bar, T_up_K, Z, Cv, valve_travel)
        Q_kgs = self.flow_kg_per_s(P_up_bar, P_down_bar, T_up_K, Z, Cv, valve_travel)
        T_out = self.temperature_out(T_up_K, P_up_bar, P_down_bar)
        choked = self.is_choked(P_up_bar, P_down_bar)
        Y = self.expansion_factor_Y(P_up_bar, P_down_bar)

        return {
            "flow_std_m3_per_h": Q_m3h,
            "flow_kg_per_s": Q_kgs,
            "T_downstream_K": T_out,
            "is_choked": choked,
            "expansion_factor_Y": Y,
        }
