"""
EC216 — Thermoelectric Generator (TEG) — F1a ZT Model

Efficiency and power from ZT figure of merit:
  eta_Carnot = 1 - T_cold/T_hot                              [K basis]
  eta = eta_Carnot * (sqrt(1+ZT_avg) - 1) / (sqrt(1+ZT_avg) + T_cold/T_hot)
  Q_hot = K * dT  (heat input via thermal conduction, simplified)
         K = k_eff * module_area / module_thickness   [W/K]
  P_max = alpha^2 * dT^2 / (4*R)   [matched load condition]
  heat_input = P_max / eta            [W]
  voltage = alpha * dT / 2           [V at matched load, N_couples series]

References:
    Rowe, D.M. (ed.) (2006). Thermoelectrics Handbook. CRC Press.
    Snyder, G.J. & Toberer, E.S. (2008). Nature Materials, 7, 105-114.
"""

import numpy as np


class TEGF1a:
    """Thermoelectric generator — ZT-based efficiency and power model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.ZT = u["ZT"]["value"]
        self.N = u["N_couples"]["value"]
        self.area = u["module_area"]["value"]           # m2
        self.thickness = u["module_thickness"]["value"] # m
        self.k_eff = u["k_eff"]["value"]               # W/(m*K)
        self.alpha = u["alpha_seebeck"]["value"]        # V/K (per couple)
        self.R_int = u["R_internal"]["value"]           # ohm (per module)
        # Thermal conductance of module [W/K]
        self.K_module = self.k_eff * self.area / self.thickness

    def efficiency(self, T_hot_c, T_cold_c):
        """Thermoelectric efficiency from ZT figure of merit.

        eta = eta_Carnot * (sqrt(1+ZT) - 1) / (sqrt(1+ZT) + T_cold/T_hot)
        """
        T_h = np.asarray(T_hot_c, dtype=float) + 273.15
        T_c = np.asarray(T_cold_c, dtype=float) + 273.15
        eta_c = 1.0 - T_c / T_h
        sqrt_term = np.sqrt(1.0 + self.ZT)
        eta = eta_c * (sqrt_term - 1.0) / (sqrt_term + T_c / T_h)
        return np.clip(eta, 0.0, 0.5)

    def power_max(self, T_hot_c, T_cold_c):
        """Maximum power output [W] at matched load.

        P_max = alpha^2 * N^2 * dT^2 / (4 * R_int)
        alpha_total = alpha * N (couples in series)
        """
        dT = np.asarray(T_hot_c, dtype=float) - np.asarray(T_cold_c, dtype=float)
        dT = np.clip(dT, 0.0, None)
        alpha_total = self.alpha * self.N
        return alpha_total**2 * dT**2 / (4.0 * self.R_int)

    def heat_input(self, T_hot_c, T_cold_c):
        """Heat input to hot side [W]."""
        P = self.power_max(T_hot_c, T_cold_c)
        eta = self.efficiency(T_hot_c, T_cold_c)
        # Avoid division by zero for eta ~0
        safe_eta = np.where(eta > 1e-6, eta, 1e-6)
        return P / safe_eta

    def voltage(self, T_hot_c, T_cold_c):
        """Open-circuit voltage [V] at matched load (half of OCV)."""
        dT = np.asarray(T_hot_c, dtype=float) - np.asarray(T_cold_c, dtype=float)
        dT = np.clip(dT, 0.0, None)
        alpha_total = self.alpha * self.N
        return 0.5 * alpha_total * dT  # voltage at matched load = OCV/2

    def compute(self, T_hot_c, T_cold_c):
        """Full compute for given temperatures.

        Returns
        -------
        dict: efficiency, power_w, heat_input_w, voltage_v
        """
        eta = self.efficiency(T_hot_c, T_cold_c)
        P = self.power_max(T_hot_c, T_cold_c)
        Q = self.heat_input(T_hot_c, T_cold_c)
        V = self.voltage(T_hot_c, T_cold_c)
        return {
            "efficiency": eta,
            "power_w": P,
            "heat_input_w": Q,
            "voltage_v": V,
        }
