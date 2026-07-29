"""
EC217 — Thermoelectric Cooler (TEC) — F1a ZT Efficiency / COP Model

Peltier cooler using Bi2Te3 modules. Semi-empirical model combining:
- Ideal COP scaled by ZT figure of merit
- Physical Peltier-Joule-Fourier balance for Q_cool

Physical model per module:
    Q_cool = N * [alpha * Tc * I - 0.5 * I^2 * R - K * (Th - Tc)]
      alpha: Seebeck coefficient [V/K]
      Tc:    cold side temperature [K]
      I:     operating current [A]
      R:     module electrical resistance [ohm]
      K:     module thermal conductance [W/K]
      Th:    hot side temperature [K]

Ideal COP (Carnot):
    COP_Carnot = Tc / (Th - Tc)

ZT-limited COP (Altenkirch formula):
    COP_ZT = COP_Carnot * eta_zt
    eta_zt = (sqrt(1 + ZT) - Th/Tc) / (sqrt(1 + ZT) + 1)
    where ZT = Z * T_mean = Z * (Tc + Th) / 2

Input power:
    W_in = Q_cool / COP_ZT   (from COP definition)
    Also directly: W_in = N * [alpha * (Th - Tc) * I + I^2 * R]

References:
    Rowe, D.M. (Ed.) (2006). Thermoelectrics Handbook: Macro to Nano. CRC Press.
    Goldsmid, H.J. (2010). Introduction to Thermoelectricity. Springer.
"""

import numpy as np


class TECF1a:
    """TEC Peltier cooler: ZT-limited COP and physical Q_cool model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.ZT = u["ZT"]["value"]
        self.N = u["N_couples"]["value"]
        self.alpha = u["alpha_Seebeck"]["value"]  # V/K per couple
        self.R = u["R_module"]["value"]            # ohm (total module)
        self.K = u["K_module"]["value"]            # W/K (total module)
        self.T_ref = u["T_ref"]["value"]           # K
        self.I_max = u["I_max"]["value"]

    def ZT_at_T(self, Tc_K, Th_K):
        """ZT evaluated at mean temperature."""
        T_mean = np.asarray((Tc_K + Th_K) / 2.0, dtype=float)
        # ZT_ref defined at T_ref; scale by T_mean/T_ref (simple linear approximation)
        return self.ZT * T_mean / self.T_ref

    def COP_carnot(self, Tc_K, Th_K):
        """Carnot (ideal) COP = Tc / (Th - Tc)."""
        Tc = np.asarray(Tc_K, dtype=float)
        Th = np.asarray(Th_K, dtype=float)
        dT = Th - Tc
        return np.where(dT > 0.0, Tc / dT, 0.0)

    def eta_zt(self, Tc_K, Th_K):
        """ZT efficiency factor (Altenkirch formula)."""
        Tc = np.asarray(Tc_K, dtype=float)
        Th = np.asarray(Th_K, dtype=float)
        zt = self.ZT_at_T(Tc, Th)
        sqrt_term = np.sqrt(1.0 + zt)
        num = sqrt_term - Th / Tc
        den = sqrt_term + 1.0
        return np.where(den > 0.0, num / den, 0.0)

    def COP(self, Tc_K, Th_K):
        """ZT-limited COP = COP_carnot * eta_zt."""
        cop_c = self.COP_carnot(Tc_K, Th_K)
        eta = self.eta_zt(Tc_K, Th_K)
        return np.maximum(cop_c * eta, 0.0)

    def Q_cool_physical(self, Tc_K, Th_K, I_A):
        """
        Physical cooling power [W] per Peltier-Joule-Fourier balance.
        Q_cool = N * [alpha * Tc * I - 0.5 * I^2 * R/N - K/N * (Th - Tc)]
        (R and K are module totals; per-couple values: R/N, K/N)
        """
        Tc = np.asarray(Tc_K, dtype=float)
        Th = np.asarray(Th_K, dtype=float)
        I = np.asarray(I_A, dtype=float)
        dT = Th - Tc
        Q = self.N * (self.alpha * Tc * I - 0.5 * I**2 * self.R / self.N
                      - self.K / self.N * dT)
        return Q  # can be negative (no cooling achieved)

    def W_input(self, Tc_K, Th_K, I_A):
        """Electrical input power [W] = N * [alpha * dT * I + I^2 * R/N]."""
        Tc = np.asarray(Tc_K, dtype=float)
        Th = np.asarray(Th_K, dtype=float)
        I = np.asarray(I_A, dtype=float)
        dT = Th - Tc
        return self.N * (self.alpha * dT * I + I**2 * self.R / self.N)

    def COP_physical(self, Tc_K, Th_K, I_A):
        """Physical COP = Q_cool / W_in."""
        Q = self.Q_cool_physical(Tc_K, Th_K, I_A)
        W = self.W_input(Tc_K, Th_K, I_A)
        return np.where(W > 0.0, Q / W, 0.0)

    def I_optimal(self, Tc_K, Th_K):
        """
        Current [A] that maximizes Q_cool (dQ/dI = 0):
        I_opt = alpha * Tc * N / R
        """
        Tc = np.asarray(Tc_K, dtype=float)
        I_opt = self.alpha * Tc * self.N / self.R
        return np.clip(I_opt, 0.0, self.I_max)
