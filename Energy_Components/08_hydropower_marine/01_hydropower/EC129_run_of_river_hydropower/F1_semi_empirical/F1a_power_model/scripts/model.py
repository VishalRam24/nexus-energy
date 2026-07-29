"""
EC129 — Run-of-River Hydropower — F1a Power Model

P = eta_overall * rho * g * Q * H_net / 1000  (kW)
H_net = H_gross * (1 - f_loss)
eta_overall = eta_turbine(Q/Q_design) * eta_generator
eta_turbine(q) = eta_peak * (1 - k*(q-1)^2)

Key differences from EC128 (large dam):
  - Low head: 2-20 m (no large reservoir, run-of-river)
  - Higher flow rates for same power
  - Kaplan/propeller turbines dominate; slightly lower eta_peak ~0.85-0.90
  - Head loss fraction explicit (short penstock)
  - Flow-duration curve dependency noted (not modelled here at F1a)

Reference:
    Penche (1998), "Layman's Guidebook on How to Develop a Small Hydro Site", EC DGXVII.
    Gordon (2001), "Hydraulic turbine efficiency", Can. J. Civ. Eng.
"""

import numpy as np


class RunOfRiverF1a:
    """Run-of-river hydropower plant — power as a function of flow and gross head."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.H_design = u["H_design"]["value"]          # m  (net)
        self.Q_design = u["Q_design"]["value"]          # m3/s
        self.P_rated = u["P_rated"]["value"]            # kW
        self.eta_peak = u["eta_peak"]["value"]
        self.eta_gen = u["eta_generator"]["value"]
        self.k = u["k_efficiency"]["value"]
        self.q_min = u["q_min"]["value"]
        self.q_max = u["q_max"]["value"]
        self.f_loss = u["head_loss_fraction"]["value"]  # dimensionless
        self.rho = u["rho"]["value"]                    # kg/m3
        self.g = u["g"]["value"]                        # m/s2

    def net_head(self, gross_head_m):
        """Net head after penstock losses."""
        return np.asarray(gross_head_m, dtype=float) * (1.0 - self.f_loss)

    def turbine_efficiency(self, Q_m3s):
        """Turbine efficiency vs flow (parabolic, Kaplan/Francis RoR)."""
        Q = np.asarray(Q_m3s, dtype=float)
        q = Q / self.Q_design
        eta_t = self.eta_peak * (1.0 - self.k * (q - 1.0) ** 2)
        eta_t = np.where((q < self.q_min) | (q > self.q_max), 0.0, eta_t)
        return np.clip(eta_t, 0.0, self.eta_peak)

    def overall_efficiency(self, Q_m3s):
        """Overall plant efficiency = eta_turbine * eta_generator."""
        eta_t = self.turbine_efficiency(Q_m3s)
        return np.where(eta_t > 0.0, eta_t * self.eta_gen, 0.0)

    def power_kw(self, Q_m3s, gross_head_m):
        """Electrical output power in kW."""
        Q = np.asarray(Q_m3s, dtype=float)
        H_net = self.net_head(gross_head_m)
        eta = self.overall_efficiency(Q)
        P = eta * self.rho * self.g * Q * H_net / 1000.0   # kW
        return np.clip(P, 0.0, self.P_rated)

    def capacity_factor(self, Q_m3s, gross_head_m):
        """Capacity factor = P / P_rated."""
        return self.power_kw(Q_m3s, gross_head_m) / self.P_rated
