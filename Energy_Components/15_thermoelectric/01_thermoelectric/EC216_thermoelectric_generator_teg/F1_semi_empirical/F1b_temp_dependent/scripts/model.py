"""
EC216 — Thermoelectric Generator (TEG) — F1b Temperature-Dependent Properties Model

Extends F1a constant-ZT model with temperature-dependent Bi2Te3 material properties:
  alpha(T) = alpha0 * (1 + a1*(T-T0) + a2*(T-T0)^2)   Seebeck coefficient
  k(T)     = k0 * (1 + b1*(T-T0))                      Thermal conductivity
  sigma(T) = sigma0 * (1 + c1*(T-T0))                   Electrical conductivity
  ZT(T)    = alpha(T)^2 * sigma(T) * T / k(T)           Figure of merit

Average ZT across temperature gradient computed by integration.
Efficiency uses the standard formula with ZT_avg.

Reference:
    Rowe, D.M. (ed.) (2006). Thermoelectrics Handbook. CRC Press.
    Snyder, G.J. & Toberer, E.S. (2008). Nature Materials, 7, 105-114.
"""

import numpy as np


class TEGF1b:
    """TEG with temperature-dependent Bi2Te3 material properties."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.alpha0 = u["alpha0"]["value"]       # V/K
        self.k0 = u["k0"]["value"]               # W/(m*K)
        self.sigma0 = u["sigma_e0"]["value"]      # S/m
        self.T0 = u["T0_K"]["value"]             # K
        self.a1 = u["a1"]["value"]               # 1/K
        self.a2 = u["a2"]["value"]               # 1/K^2
        self.b1 = u["b1"]["value"]               # 1/K
        self.c1 = u["c1"]["value"]               # 1/K
        self.N = u["n_couples"]["value"]
        self.A_elem = u["A_element_m2"]["value"]
        self.L_elem = u["L_element_m"]["value"]
        self.module_area = u["module_area"]["value"]
        self.r_contact = u["contact_resistance_fraction"]["value"]

    def alpha(self, T):
        """Seebeck coefficient (V/K) at temperature T (K)."""
        T = np.asarray(T, dtype=float)
        dT = T - self.T0
        return self.alpha0 * (1.0 + self.a1 * dT + self.a2 * dT ** 2)

    def k_thermal(self, T):
        """Thermal conductivity (W/(m*K)) at temperature T (K)."""
        T = np.asarray(T, dtype=float)
        dT = T - self.T0
        return self.k0 * (1.0 + self.b1 * dT)

    def sigma_electrical(self, T):
        """Electrical conductivity (S/m) at temperature T (K)."""
        T = np.asarray(T, dtype=float)
        dT = T - self.T0
        return self.sigma0 * (1.0 + self.c1 * dT)

    def zt_local(self, T):
        """Local ZT at temperature T (K).
        ZT = alpha^2 * sigma * T / k
        """
        T = np.asarray(T, dtype=float)
        a = self.alpha(T)
        k = self.k_thermal(T)
        s = self.sigma_electrical(T)
        return a ** 2 * s * T / (k + 1e-12)

    def zt_average(self, T_hot_K, T_cold_K):
        """Average ZT across temperature gradient using trapezoidal integration.
        ZT_avg = (1/(T_h-T_c)) * integral_{T_c}^{T_h} ZT(T) dT
        """
        T_h = np.asarray(T_hot_K, dtype=float)
        T_c = np.asarray(T_cold_K, dtype=float)

        # Use scalar path for scalar inputs, vectorized for arrays
        if T_h.ndim == 0 and T_c.ndim == 0:
            T_h_val = float(T_h)
            T_c_val = float(T_c)
            n_pts = 50
            T_arr = np.linspace(T_c_val, T_h_val, n_pts)
            zt_arr = self.zt_local(T_arr)
            # np.trapezoid for NumPy >=2.0, fallback to np.trapz for older
            try:
                zt_avg = np.trapezoid(zt_arr, T_arr) / (T_h_val - T_c_val + 1e-12)
            except AttributeError:
                zt_avg = np.trapz(zt_arr, T_arr) / (T_h_val - T_c_val + 1e-12)
            return float(zt_avg)
        else:
            # For array inputs, compute at average temperature as approximation
            T_avg = (T_h + T_c) / 2.0
            return self.zt_local(T_avg)

    def efficiency(self, T_hot_K, T_cold_K):
        """Thermoelectric efficiency with temperature-dependent ZT.
        eta = eta_Carnot * (sqrt(1+ZT_avg) - 1) / (sqrt(1+ZT_avg) + T_c/T_h)
        """
        T_h = np.asarray(T_hot_K, dtype=float)
        T_c = np.asarray(T_cold_K, dtype=float)

        eta_carnot = 1.0 - T_c / T_h
        ZT = self.zt_average(T_h, T_c)
        ZT = np.asarray(ZT, dtype=float)

        sqrt_zt = np.sqrt(1.0 + ZT)
        eta = eta_carnot * (sqrt_zt - 1.0) / (sqrt_zt + T_c / T_h)
        return np.clip(eta, 0.0, 0.5)

    def _module_resistance(self, T_avg):
        """Total module electrical resistance (ohm) at average temperature.
        R = 2*N * (L / (sigma * A)) * (1 + r_contact)
        Factor 2: each couple has p-type and n-type legs.
        """
        sigma = self.sigma_electrical(T_avg)
        R_elem = self.L_elem / (sigma * self.A_elem + 1e-12)
        return 2.0 * self.N * R_elem * (1.0 + self.r_contact)

    def _module_thermal_conductance(self, T_avg):
        """Module thermal conductance (W/K) at average temperature.
        K = 2*N * k * A / L
        """
        k = self.k_thermal(T_avg)
        return 2.0 * self.N * k * self.A_elem / self.L_elem

    def power_density_w_cm2(self, T_hot_K, T_cold_K):
        """Power output per unit module area (W/cm2) at matched load.
        P_max = alpha_total^2 * dT^2 / (4 * R_int)
        power_density = P_max / module_area (converted to cm2)
        """
        T_h = np.asarray(T_hot_K, dtype=float)
        T_c = np.asarray(T_cold_K, dtype=float)
        T_avg = (T_h + T_c) / 2.0
        dT = T_h - T_c

        alpha_avg = self.alpha(T_avg)
        alpha_total = alpha_avg * self.N
        R_int = self._module_resistance(T_avg)

        P_max = alpha_total ** 2 * dT ** 2 / (4.0 * R_int + 1e-12)
        # W to W/cm2: module_area in m2 -> cm2: *1e4
        return P_max / (self.module_area * 1e4)

    def voltage_V(self, T_hot_K, T_cold_K):
        """Voltage at matched load (V) = V_oc / 2.
        V_oc = alpha_total * dT
        """
        T_h = np.asarray(T_hot_K, dtype=float)
        T_c = np.asarray(T_cold_K, dtype=float)
        T_avg = (T_h + T_c) / 2.0
        dT = T_h - T_c

        alpha_avg = self.alpha(T_avg)
        alpha_total = alpha_avg * self.N
        return 0.5 * alpha_total * dT

    def compute(self, T_hot_K, T_cold_K):
        """Full computation returning all outputs."""
        eta = self.efficiency(T_hot_K, T_cold_K)
        pd = self.power_density_w_cm2(T_hot_K, T_cold_K)
        zt = self.zt_average(T_hot_K, T_cold_K)
        V = self.voltage_V(T_hot_K, T_cold_K)

        return {
            "efficiency": eta,
            "power_density_w_cm2": pd,
            "zt_average": zt,
            "voltage_V": V,
        }
