"""
EC217 — Thermoelectric Cooler (TEC) — F1b Temperature-Dependent Properties Model

Extends F1a constant-ZT TEC model with temperature-dependent Bi2Te3 material properties,
contact resistance, Peltier effect, and Thomson heat correction.

Material property fits (same as EC216 F1b — TEC and TEG use identical Bi2Te3 material):
    alpha(T) = alpha0 * (1 + a1*(T-T0) + a2*(T-T0)^2)   [V/K] Seebeck coefficient
    k(T)     = k0 * (1 + b1*(T-T0))                      [W/(m*K)] Thermal conductivity
    sigma(T) = sigma0 * (1 + c1*(T-T0))                   [S/m] Electrical conductivity

TEC operating equations (current-controlled):
    Q_cold   = alpha*I*T_c - 0.5*I^2*R - K*(T_h - T_c)  [W]  cooling power (Peltier - Joule/2 - conduction)
    W_input  = alpha*I*(T_h - T_c) + I^2*R               [W]  input electrical power
    COP      = Q_cold / W_input
    T_min    = T_hot - alpha^2*T_hot^2 / (2*k*R_e)       [K]  minimum achievable cold-side temp (Ioffe formula)
    I_opt    = alpha*T_c / R_e                            [A]  current for max COP (from differentiation)

Thomson effect correction (1st order):
    Q_Thomson = tau * I * (T_h - T_c)   where tau ~ T * d(alpha)/dT

Contact resistance included: R_total = R_ideal * (1 + r_contact)

References:
    Rowe, D.M. (ed.) (2006). Thermoelectrics Handbook. CRC Press.
    Goldsmid, H.J. (1986). Electronic Refrigeration. Pion Ltd.
    Ioffe, A.F. (1957). Semiconductor Thermoelements and Thermoelectric Cooling. Infosearch.
"""

import numpy as np


class TECF1b:
    """TEC with temperature-dependent Bi2Te3 material properties + contact resistance + Thomson."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.alpha0 = u["alpha0"]["value"]           # V/K
        self.k0 = u["k0"]["value"]                   # W/(m*K)
        self.sigma0 = u["sigma_e0"]["value"]          # S/m
        self.T0 = u["T0_K"]["value"]                 # K
        self.a1 = u["a1"]["value"]                   # 1/K
        self.a2 = u["a2"]["value"]                   # 1/K^2
        self.b1 = u["b1"]["value"]                   # 1/K
        self.c1 = u["c1"]["value"]                   # 1/K
        self.N = u["n_couples"]["value"]
        self.A_elem = u["A_element_m2"]["value"]
        self.L_elem = u["L_element_m"]["value"]
        self.module_area = u["module_area"]["value"]
        self.r_contact = u["contact_resistance_fraction"]["value"]

    # ------------------------------------------------------------------ #
    # Material property functions                                          #
    # ------------------------------------------------------------------ #

    def alpha(self, T):
        """Seebeck coefficient [V/K] at temperature T [K]."""
        T = np.asarray(T, dtype=float)
        dT = T - self.T0
        return self.alpha0 * (1.0 + self.a1 * dT + self.a2 * dT ** 2)

    def k_thermal(self, T):
        """Thermal conductivity [W/(m*K)] at temperature T [K]."""
        T = np.asarray(T, dtype=float)
        dT = T - self.T0
        return self.k0 * (1.0 + self.b1 * dT)

    def sigma_electrical(self, T):
        """Electrical conductivity [S/m] at temperature T [K]."""
        T = np.asarray(T, dtype=float)
        dT = T - self.T0
        return self.sigma0 * (1.0 + self.c1 * dT)

    def zt_local(self, T):
        """Local figure of merit ZT = alpha^2 * sigma * T / k."""
        T = np.asarray(T, dtype=float)
        a = self.alpha(T)
        k = self.k_thermal(T)
        s = self.sigma_electrical(T)
        return a ** 2 * s * T / (k + 1e-12)

    # ------------------------------------------------------------------ #
    # Module-level derived quantities                                      #
    # ------------------------------------------------------------------ #

    def _module_resistance(self, T_avg):
        """Total module electrical resistance [ohm] at average temperature.
        R = 2*N * (L / (sigma * A)) * (1 + r_contact)
        """
        sigma = self.sigma_electrical(T_avg)
        R_elem = self.L_elem / (sigma * self.A_elem + 1e-12)
        return 2.0 * self.N * R_elem * (1.0 + self.r_contact)

    def _module_thermal_conductance(self, T_avg):
        """Module thermal conductance [W/K] at average temperature.
        K = 2*N * k * A / L
        """
        k = self.k_thermal(T_avg)
        return 2.0 * self.N * k * self.A_elem / self.L_elem

    def _alpha_module(self, T_avg):
        """Total module Seebeck coefficient [V/K].
        alpha_module = N * alpha(T_avg)
        (Peltier: both p and n arms contribute to same sign Peltier effect)
        """
        return self.N * self.alpha(T_avg)

    def _thomson_heat(self, T_cold_K, T_hot_K, I):
        """Thomson heat [W] — correction for non-constant Seebeck coefficient.
        Q_T = tau * I * (T_h - T_c)
        where tau = T * d(alpha)/dT = T * alpha0 * a1 (at T_avg)
        The Thomson effect reduces cold-side cooling slightly (approx ±).
        """
        T_avg = (T_cold_K + T_hot_K) / 2.0
        # d(alpha_module)/dT = N * alpha0 * (a1 + 2*a2*(T-T0))
        dalpha_dT = self.N * self.alpha0 * (self.a1 + 2.0 * self.a2 * (T_avg - self.T0))
        tau = T_avg * dalpha_dT  # Thomson coefficient [V/K]
        dT = T_hot_K - T_cold_K
        return tau * np.asarray(I, dtype=float) * dT

    # ------------------------------------------------------------------ #
    # Core TEC computation                                                 #
    # ------------------------------------------------------------------ #

    def compute(self, T_cold_K, T_hot_K, I_A):
        """
        Compute TEC performance at given operating conditions.

        Parameters
        ----------
        T_cold_K : float or array — cold-side temperature [K]
        T_hot_K  : float or array — hot-side temperature [K]
        I_A      : float or array — drive current [A]

        Returns
        -------
        dict: Q_cold_W, Q_hot_W, W_input_W, COP, COP_max_theoretical,
              T_min_achievable_K, ZT_avg, V_module_V
        """
        T_c = np.asarray(T_cold_K, dtype=float)
        T_h = np.asarray(T_hot_K, dtype=float)
        I = np.asarray(I_A, dtype=float)
        T_avg = (T_c + T_h) / 2.0

        # Temperature-dependent module parameters at average T
        alpha_m = self._alpha_module(T_avg)
        R_e = self._module_resistance(T_avg)
        K_th = self._module_thermal_conductance(T_avg)

        # Thomson heat
        Q_thomson = self._thomson_heat(T_c, T_h, I)

        # Cooling power (modified Peltier equation with Thomson correction)
        # Q_cold = alpha*I*T_c - 0.5*I^2*R - K*(T_h-T_c) - 0.5*Q_thomson
        # Thomson distributes as +0.5*tau*I*dT at cold junction
        Q_cold = (alpha_m * I * T_c
                  - 0.5 * I ** 2 * R_e
                  - K_th * (T_h - T_c)
                  - 0.5 * Q_thomson)

        # Electrical input power
        # W = alpha_m * I * (T_h - T_c) + I^2 * R_e
        W_input = alpha_m * I * (T_h - T_c) + I ** 2 * R_e
        W_input = np.maximum(W_input, 1e-12)

        # Heat rejected at hot side (energy balance)
        Q_hot = Q_cold + W_input

        # COP
        Q_cold_safe = np.where(Q_cold > 0, Q_cold, 0.0)
        COP = Q_cold_safe / W_input

        # Theoretical max COP (Ioffe, constant-property approximation at T_avg)
        ZT_avg = self.zt_local(T_avg)
        sqrt_zt = np.sqrt(1.0 + ZT_avg)
        # COP_max = T_c / (T_h - T_c) * (sqrt_zt - T_h/T_c) / (sqrt_zt + 1)
        dT_safe = np.maximum(T_h - T_c, 1e-3)
        COP_max = (T_c / dT_safe) * (sqrt_zt - T_h / T_c) / (sqrt_zt + 1.0)
        COP_max = np.clip(COP_max, 0.0, 10.0)

        # Minimum achievable temperature (Ioffe formula):
        # T_c_min = T_h - T_h * (1 - sqrt(1 + 2*ZT_avg)) / (ZT_avg)
        # Simplified: T_c_min = T_h * (1 - ZT_avg/2) for ZT << 1
        # Full Ioffe: T_c_min = T_h - ZT_avg/2 * T_h (approximate)
        # More accurate: solve d(Q_cold)/dI=0 for I_opt, then set Q_cold=0
        I_opt = alpha_m * T_c / (R_e + 1e-12)
        Q_cold_at_iopt = (alpha_m * I_opt * T_c
                          - 0.5 * I_opt ** 2 * R_e
                          - K_th * (T_h - T_c))
        # T_min: temperature where Q_cold=0 at optimal current
        # T_c_min = T_h - 0.5 * ZT_avg * T_h (linearised, valid for ZT~1)
        T_min = T_h * (1.0 - 0.5 * ZT_avg / (1.0 + ZT_avg))
        T_min = np.maximum(T_min, 50.0)

        # Module terminal voltage
        V_module = alpha_m * (T_h - T_c) + I * R_e

        return {
            "Q_cold_W": Q_cold,
            "Q_hot_W": Q_hot,
            "W_input_W": W_input,
            "COP": COP,
            "COP_max_theoretical": COP_max,
            "T_min_achievable_K": T_min,
            "ZT_avg": ZT_avg,
            "V_module_V": V_module,
        }

    def compute_optimal_current(self, T_cold_K, T_hot_K):
        """Return optimal current for maximum Q_cold at given T boundary."""
        T_c = np.asarray(T_cold_K, dtype=float)
        T_h = np.asarray(T_hot_K, dtype=float)
        T_avg = (T_c + T_h) / 2.0
        alpha_m = self._alpha_module(T_avg)
        R_e = self._module_resistance(T_avg)
        K_th = self._module_thermal_conductance(T_avg)
        # dQ_cold/dI = alpha_m*T_c - I*R_e = 0 -> I_opt = alpha_m*T_c/R_e
        I_opt = alpha_m * T_c / (R_e + 1e-12)
        return float(np.atleast_1d(I_opt)[0])
