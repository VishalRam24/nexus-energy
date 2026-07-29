"""
EC095 — Thermoelectric Cooler (Peltier) — F1a COP Curve Model

Semi-empirical lumped Peltier module stack.  Per-module governing
equations (Goldsmid; Rowe Handbook):

    Q_c = alpha * I * T_c - 0.5 * I**2 * R - K * (T_h - T_c)
    W_in = alpha * I * (T_h - T_c) + I**2 * R
    Q_h = Q_c + W_in
    COP_c = Q_c / W_in            (only when W_in > 0)

with:
    alpha [V/K] : effective Seebeck coefficient of one module
    R     [Ohm]: total electrical resistance of one module
    K     [W/K]: total thermal conductance of one module
    I     [A]  : module current (the modules are in series; same I)
    T_c, T_h [K]: cold-side and hot-side absolute temperatures.

The stack of N modules has:
    Q_c_stack = N * Q_c_module
    W_stack   = N * W_in_module

References:
    Goldsmid, H.J. (2010). Introduction to Thermoelectricity, Springer.
    Rowe, D.M. (Ed.) (2006). CRC Handbook of Thermoelectrics.
    Riffat, S.B., Ma, X. (2003). Appl. Thermal Eng. 23, 913-935.
"""

import numpy as np


class PeltierTECF1a:
    """Multi-module Peltier (TEC) cooler — Q_c, W_in, COP_c from first principles."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.N      = int(u["n_modules"]["value"])
        self.alpha  = u["alpha_module"]["value"]
        self.R      = u["R_module"]["value"]
        self.K      = u["K_module"]["value"]
        self.I_max  = u["I_max"]["value"]
        self.aux_W  = u["auxiliary_power"]["value"]

    # ------------------------------------------------------------------
    def cooling_power(self, current_a, T_cold_c, T_hot_c):
        """Cold-side heat absorption Q_c (per stack) in W. Negative means
        heat would flow backwards (i.e. inadequate current); we clip at 0."""
        I  = np.asarray(current_a, dtype=float)
        Tc = np.asarray(T_cold_c, dtype=float) + 273.15
        Th = np.asarray(T_hot_c,  dtype=float) + 273.15
        Q_c_mod = self.alpha * I * Tc - 0.5 * I * I * self.R - self.K * (Th - Tc)
        return np.maximum(self.N * Q_c_mod, 0.0)

    def electrical_input(self, current_a, T_cold_c, T_hot_c):
        """Total electrical input W_in (stack) in W."""
        I  = np.asarray(current_a, dtype=float)
        Tc = np.asarray(T_cold_c, dtype=float) + 273.15
        Th = np.asarray(T_hot_c,  dtype=float) + 273.15
        W_mod = self.alpha * I * (Th - Tc) + I * I * self.R
        return self.N * W_mod + self.aux_W

    def heat_rejection(self, current_a, T_cold_c, T_hot_c):
        """Hot-side heat rejection Q_h = Q_c + W_in (stack) in W."""
        return (self.cooling_power(current_a, T_cold_c, T_hot_c)
                + self.electrical_input(current_a, T_cold_c, T_hot_c))

    def cop(self, current_a, T_cold_c, T_hot_c):
        """COP_c = Q_c / W_in.  Returns 0 when no usable cooling."""
        Qc = self.cooling_power(current_a, T_cold_c, T_hot_c)
        W  = self.electrical_input(current_a, T_cold_c, T_hot_c)
        cop = np.where(W > 1e-9, Qc / np.where(W > 1e-9, W, 1.0), 0.0)
        return np.clip(cop, 0.0, 5.0)

    def optimum_current(self, T_cold_c, T_hot_c):
        """Current that maximises Q_c at given (T_c, T_h):
        d Q_c / dI = alpha*T_c - I*R = 0  =>  I_opt = alpha*T_c/R."""
        Tc = np.asarray(T_cold_c, dtype=float) + 273.15
        return np.minimum(self.alpha * Tc / self.R, self.I_max)
