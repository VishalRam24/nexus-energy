"""
EC128 — Conventional Hydroelectric Dam — F1a Power Model

P = eta_overall * rho * g * Q * H / 1000  (kW)
eta_overall = eta_turbine(Q/Q_design) * eta_generator
eta_turbine(q) = eta_peak * (1 - k*(q-1)^2)  where q = Q/Q_design

Reference:
    Dixon & Hall (2014), "Fluid Mechanics and Thermodynamics of Turbomachinery",
    7th ed., Butterworth-Heinemann, ch. 9.
"""

import numpy as np


class HydroelectricDamF1a:
    """Francis turbine hydroelectric dam — power as a function of flow and head."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.H_design = u["H_design"]["value"]       # m
        self.Q_design = u["Q_design"]["value"]       # m3/s
        self.P_rated = u["P_rated"]["value"]         # kW
        self.eta_peak = u["eta_peak"]["value"]
        self.eta_gen = u["eta_generator"]["value"]
        self.k = u["k_efficiency"]["value"]
        self.q_min = u["q_min"]["value"]
        self.q_max = u["q_max"]["value"]
        self.rho = u["rho"]["value"]                 # kg/m3
        self.g = u["g"]["value"]                     # m/s2

    def turbine_efficiency(self, Q_m3s):
        """Turbine efficiency as a function of flow rate (Francis turbine curve)."""
        Q = np.asarray(Q_m3s, dtype=float)
        q = Q / self.Q_design                        # normalized flow
        # Parabolic efficiency curve, peak at q=1
        eta_t = self.eta_peak * (1.0 - self.k * (q - 1.0) ** 2)
        # Clamp: zero efficiency below q_min (cut-in) or above q_max (runaway)
        eta_t = np.where((q < self.q_min) | (q > self.q_max), 0.0, eta_t)
        return np.clip(eta_t, 0.0, self.eta_peak)

    def overall_efficiency(self, Q_m3s):
        """Overall plant efficiency = eta_turbine * eta_generator."""
        eta_t = self.turbine_efficiency(Q_m3s)
        # Generator efficiency applied only when turbine is producing
        eta_overall = np.where(eta_t > 0.0, eta_t * self.eta_gen, 0.0)
        return eta_overall

    def power_kw(self, Q_m3s, head_m):
        """Electrical output power in kW."""
        Q = np.asarray(Q_m3s, dtype=float)
        H = np.asarray(head_m, dtype=float)
        eta = self.overall_efficiency(Q)
        P = eta * self.rho * self.g * Q * H / 1000.0   # kW
        return np.clip(P, 0.0, self.P_rated)

    def capacity_factor(self, Q_m3s, head_m):
        """Capacity factor = P / P_rated."""
        P = self.power_kw(Q_m3s, head_m)
        return P / self.P_rated
