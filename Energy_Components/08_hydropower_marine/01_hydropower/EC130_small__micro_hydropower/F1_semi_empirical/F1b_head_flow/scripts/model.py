"""
EC130 — Small/Micro Hydropower — F1b Head-Flow Model

Extends F1a with:
1. 2D efficiency hill chart: eta(q, h) = eta_peak * (1-k_q*(q-1)^2) * (1-k_h*(h-1)^2)
   Each turbine type has its own k_q, k_h, q_min, q_max, h_min, h_max.

2. Variable head losses: f_loss(Q) = f_loss_design * (Q/Q_design)^2

3. Cavitation check:
   sigma_plant = (H_atm - H_vapor - H_draft) / H_net
   If sigma_plant < sigma_c → efficiency derated

4. Water temperature density correction (same polynomial as EC129)

5. Part-load (30% flow) verification: Phase 7 note — EC130 test at 70% flow hit P_rated
   cap; 30% flow used for part-load check to avoid saturation.

References:
    Penche (1998), "Layman's Guidebook on How to Develop a Small Hydro Site", EC DGXVII.
    Harvey et al. (1993), "Micro-Hydro Design Manual", IT Publications.
    Fraenkel et al. (1991), "Micro-Hydro Power", IT Publications.
    IEC 60041:1991.
"""

import numpy as np

TURBINE_PARAMS = {
    "pelton":  {
        "head_min": 100.0, "head_max": 1800.0,
        "eta_peak": 0.91,
        "k_q": 0.30, "k_h": 0.30,
        "q_min": 0.10, "q_max": 1.00,
        "h_min": 0.85, "h_max": 1.10,
        "sigma_c": 0.05,    # Pelton: low cavitation (impulse turbine)
    },
    "francis": {
        "head_min": 20.0, "head_max": 300.0,
        "eta_peak": 0.89,
        "k_q": 0.25, "k_h": 0.20,
        "q_min": 0.20, "q_max": 1.10,
        "h_min": 0.60, "h_max": 1.25,
        "sigma_c": 0.25,
    },
    "kaplan":  {
        "head_min": 2.0, "head_max": 50.0,
        "eta_peak": 0.87,
        "k_q": 0.22, "k_h": 0.18,
        "q_min": 0.15, "q_max": 1.25,
        "h_min": 0.50, "h_max": 1.30,
        "sigma_c": 0.50,
    },
}


def select_turbine(net_head_m: float) -> str:
    """Select turbine type from net head."""
    h = float(np.mean(net_head_m))
    if h > 100.0:
        return "pelton"
    elif h < 20.0:
        return "kaplan"
    return "francis"


def _water_density(T_water_C):
    T = np.asarray(T_water_C, dtype=float)
    return 999.97 - 0.05 * (T - 4.0) ** 2 / (T + 300.0)


class SmallMicroHydroF1b:
    """Small/micro hydropower — 2D hill chart, variable head losses, cavitation, temperature."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.H_design   = u["H_design"]["value"]
        self.Q_design   = u["Q_design"]["value"]
        self.P_rated    = u["P_rated"]["value"]
        self.f_loss_ref = u["head_loss_fraction"]["value"]
        self.eta_gen    = u["eta_generator"]["value"]
        self.q_min      = u["q_min"]["value"]
        self.q_max      = u["q_max"]["value"]
        self.rho_ref    = u["rho_ref"]["value"]
        self.g          = u["g"]["value"]
        self.H_draft    = u["H_draft"]["value"]
        self.k_cav      = u["k_cavitation"]["value"]
        self.Q_eco      = u["Q_eco"]["value"]

    def penstock_head_loss_fraction(self, Q_m3s):
        """Head loss fraction, proportional to Q^2."""
        q = np.clip(np.asarray(Q_m3s, dtype=float) / self.Q_design, 0.0, 2.0)
        return self.f_loss_ref * q ** 2

    def net_head(self, gross_head_m, Q_m3s):
        """Net head [m] after penstock losses."""
        H_gross = np.asarray(gross_head_m, dtype=float)
        f = self.penstock_head_loss_fraction(Q_m3s)
        return H_gross * (1.0 - f)

    def turbine_efficiency(self, Q_m3s, net_head_m, turbine_type="auto"):
        """2D hill chart efficiency (type auto-selected from H if 'auto')."""
        Q = np.asarray(Q_m3s, dtype=float)
        H = np.asarray(net_head_m, dtype=float)

        if turbine_type == "auto":
            t_name = select_turbine(float(np.mean(H)))
        else:
            t_name = turbine_type
        t = TURBINE_PARAMS[t_name]

        q = Q / self.Q_design
        h = H / self.H_design

        eta_q = 1.0 - t["k_q"] * (q - 1.0) ** 2
        eta_h = 1.0 - t["k_h"] * (h - 1.0) ** 2
        eta_t = t["eta_peak"] * eta_q * eta_h

        out_of_range = (
            (q < t["q_min"]) | (q > t["q_max"]) |
            (h < t["h_min"]) | (h > t["h_max"])
        )
        eta_t = np.where(out_of_range, 0.0, eta_t)
        return np.clip(eta_t, 0.0, t["eta_peak"])

    def cavitation_derate(self, H_net_m, turbine_type="auto"):
        """Cavitation efficiency derate factor."""
        H = np.asarray(H_net_m, dtype=float)
        if turbine_type == "auto":
            t_name = select_turbine(float(np.mean(H)))
        else:
            t_name = turbine_type
        sigma_c = TURBINE_PARAMS[t_name]["sigma_c"]
        H_safe = np.maximum(H, 0.1)
        H_atm_minus_vapor = 10.0   # m (sea level, ~20°C)
        sigma_plant = (H_atm_minus_vapor - self.H_draft) / H_safe
        excess = np.maximum(sigma_c / np.maximum(sigma_plant, 1e-9) - 1.0, 0.0)
        derate = np.maximum(1.0 - self.k_cav * excess, 0.0)
        return np.where(H <= 0, 0.0, derate)

    def overall_efficiency(self, Q_m3s, net_head_m, turbine_type="auto"):
        """Overall plant efficiency = eta_turbine * eta_gen * cavitation_derate."""
        eta_t = self.turbine_efficiency(Q_m3s, net_head_m, turbine_type)
        cav = self.cavitation_derate(net_head_m, turbine_type)
        return np.where(eta_t > 0, eta_t * self.eta_gen * cav, 0.0)

    def power_kw(self, Q_m3s, gross_head_m, turbine_type="auto", T_water_C=None):
        """
        Electrical output power [kW].
        Phase 7 note: use 30% flow for part-load test to avoid P_rated cap.
        """
        Q_total = np.asarray(Q_m3s, dtype=float)
        Q_avail = np.maximum(Q_total - self.Q_eco, 0.0)
        H_net = self.net_head(gross_head_m, Q_avail)
        rho = self.rho_ref if T_water_C is None else _water_density(T_water_C)
        eta = self.overall_efficiency(Q_avail, H_net, turbine_type)
        P = eta * rho * self.g * Q_avail * H_net / 1000.0
        return np.clip(P, 0.0, self.P_rated)

    def capacity_factor(self, Q_m3s, gross_head_m, turbine_type="auto", T_water_C=None):
        return self.power_kw(Q_m3s, gross_head_m, turbine_type, T_water_C) / self.P_rated

    @staticmethod
    def turbine_type_for_head(net_head_m: float) -> str:
        return select_turbine(net_head_m)
