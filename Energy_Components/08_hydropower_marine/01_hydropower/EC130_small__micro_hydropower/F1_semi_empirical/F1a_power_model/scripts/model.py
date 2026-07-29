"""
EC130 — Small/Micro Hydropower — F1a Power Model

P = eta_overall * rho * g * Q * H_net / 1000  (kW)
eta_overall = eta_turbine(Q/Q_design) * eta_generator

Turbine type selected automatically from net head:
  Pelton   : H_net > 100 m   (impulse, high head, low flow)
  Francis  : 20 m < H_net <= 300 m  (mixed-flow reaction)
  Kaplan   : H_net <= 50 m  (axial reaction, low head)
  (Francis preferred in overlap 20-100 m when Q moderate)

Scale: 1 kW – 10 MW
eta_peak varies by type: Pelton 0.91, Francis 0.89, Kaplan 0.87

Reference:
    Penche (1998), "Layman's Guidebook on How to Develop a Small Hydro Site", EC DGXVII.
    Harvey et al. (1993), "Micro-Hydro Design Manual", IT Publications.
    Fraenkel et al. (1991), "Micro-Hydro Power", IT Publications.
"""

import numpy as np


TURBINE_TYPES = {
    "pelton":  {"head_min": 100.0, "head_max": 1800.0, "eta_peak": 0.91, "k": 0.20},
    "francis": {"head_min":  20.0, "head_max":  300.0, "eta_peak": 0.89, "k": 0.25},
    "kaplan":  {"head_min":   2.0, "head_max":   50.0, "eta_peak": 0.87, "k": 0.22},
}


def select_turbine(net_head_m: float) -> str:
    """
    Select turbine type from net head (scalar).
    Pelton for H > 100 m; Kaplan for H < 20 m; Francis in between.
    """
    if net_head_m > 100.0:
        return "pelton"
    elif net_head_m < 20.0:
        return "kaplan"
    else:
        return "francis"


class SmallMicroHydroF1a:
    """Small/micro hydropower plant — auto turbine selection by head."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.H_design = u["H_design"]["value"]          # m (net)
        self.Q_design = u["Q_design"]["value"]          # m3/s
        self.P_rated = u["P_rated"]["value"]            # kW
        self.f_loss = u["head_loss_fraction"]["value"]
        self.eta_gen = u["eta_generator"]["value"]
        self.q_min = u["q_min"]["value"]
        self.q_max = u["q_max"]["value"]
        self.rho = u["rho"]["value"]
        self.g = u["g"]["value"]

    def turbine_efficiency(self, Q_m3s, turbine_type="auto", net_head_m=None):
        """
        Turbine efficiency.

        Args:
            Q_m3s: flow rate (scalar or array)
            turbine_type: 'pelton', 'francis', 'kaplan', or 'auto'
            net_head_m: required when turbine_type='auto' (scalar)
        """
        Q = np.asarray(Q_m3s, dtype=float)
        if turbine_type == "auto":
            if net_head_m is None:
                net_head_m = self.H_design
            t_name = select_turbine(float(np.mean(net_head_m)))
        else:
            t_name = turbine_type
        t = TURBINE_TYPES[t_name]
        q = Q / self.Q_design
        eta_t = t["eta_peak"] * (1.0 - t["k"] * (q - 1.0) ** 2)
        eta_t = np.where((q < self.q_min) | (q > self.q_max), 0.0, eta_t)
        return np.clip(eta_t, 0.0, t["eta_peak"])

    def overall_efficiency(self, Q_m3s, turbine_type="auto", net_head_m=None):
        eta_t = self.turbine_efficiency(Q_m3s, turbine_type, net_head_m)
        return np.where(eta_t > 0.0, eta_t * self.eta_gen, 0.0)

    def power_kw(self, Q_m3s, net_head_m, turbine_type="auto"):
        """Electrical output power in kW."""
        Q = np.asarray(Q_m3s, dtype=float)
        H = np.asarray(net_head_m, dtype=float)
        eta = self.overall_efficiency(Q, turbine_type, H)
        P = eta * self.rho * self.g * Q * H / 1000.0
        return np.clip(P, 0.0, self.P_rated)

    def capacity_factor(self, Q_m3s, net_head_m, turbine_type="auto"):
        return self.power_kw(Q_m3s, net_head_m, turbine_type) / self.P_rated

    @staticmethod
    def turbine_type_for_head(net_head_m: float) -> str:
        return select_turbine(net_head_m)
