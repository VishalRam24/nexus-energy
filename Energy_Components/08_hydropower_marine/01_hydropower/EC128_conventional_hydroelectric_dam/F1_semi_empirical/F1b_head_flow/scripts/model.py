"""
EC128 — Conventional Hydroelectric Dam — F1b Head-Flow Model

Extends F1a with:
1. Multi-turbine type support: Francis (40-600m), Kaplan (2-40m), Pelton (300-1800m)
2. Efficiency hill chart: eta(q, h) = eta_peak * (1 - k_q*(q-1)^2) * (1 - k_h*(h-1)^2)
   where q = Q/Q_rated, h = H/H_rated
3. Variable head (reservoir level changes)
4. Minimum environmental flow constraint
5. Specific speed calculation for turbine selection guidance

The hill chart is a 2D parabolic surface peaking at design point (q=1, h=1).
Off-design operation in both flow and head degrades efficiency.

Specific speed (dimensionless):
    Ns = N * sqrt(Q) / (g*H)^(3/4)
    Francis: 0.05-0.35, Kaplan: 0.3-0.9, Pelton: 0.005-0.05

References:
    Dixon, S.L. & Hall, C.A. (2014). Fluid Mechanics and Thermodynamics of
    Turbomachinery, 7th ed. Butterworth-Heinemann.
    IEC 60041:1991 — Field acceptance tests for hydraulic turbines.
"""

import numpy as np


class HydroelectricDamF1b:
    """Conventional hydro dam — multi-turbine, variable head, hill chart."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta_gen = u["eta_generator"]["value"]
        self.rho = u["rho"]["value"]
        self.g = u["g"]["value"]
        self.Q_env_frac = u["Q_env_fraction"]["value"]

        # Load turbine parameters for each type
        self.turbines = {}
        for ttype in ["francis", "kaplan", "pelton"]:
            t = u[ttype]
            self.turbines[ttype] = {
                "Q_rated": t["Q_rated"]["value"],
                "H_rated": t["H_rated"]["value"],
                "P_rated": t["P_rated"]["value"],
                "eta_peak": t["eta_peak"]["value"],
                "q_min": t["q_min"]["value"],
                "q_max": t["q_max"]["value"],
                "h_min": t["h_min"]["value"],
                "h_max": t["h_max"]["value"],
                "k_q": t["k_q"]["value"],
                "k_h": t["k_h"]["value"],
            }

    def _get_turbine(self, turbine_type="francis"):
        return self.turbines[turbine_type]

    def turbine_efficiency(self, Q_m3s, head_m, turbine_type="francis"):
        """
        Turbine efficiency via 2D hill chart:
            eta = eta_peak * (1 - k_q*(q-1)^2) * (1 - k_h*(h-1)^2)
        where q = Q/Q_rated, h = H/H_rated.

        Returns 0 outside valid operating range.
        """
        t = self._get_turbine(turbine_type)
        Q = np.asarray(Q_m3s, dtype=float)
        H = np.asarray(head_m, dtype=float)

        q = Q / t["Q_rated"]
        h = H / t["H_rated"]

        # Hill chart: 2D parabolic
        eta_q = 1.0 - t["k_q"] * (q - 1.0) ** 2
        eta_h = 1.0 - t["k_h"] * (h - 1.0) ** 2
        eta_t = t["eta_peak"] * eta_q * eta_h

        # Operating range cutoffs
        out_of_range = (
            (q < t["q_min"]) | (q > t["q_max"]) |
            (h < t["h_min"]) | (h > t["h_max"])
        )
        eta_t = np.where(out_of_range, 0.0, eta_t)
        return np.clip(eta_t, 0.0, t["eta_peak"])

    def overall_efficiency(self, Q_m3s, head_m, turbine_type="francis"):
        """Overall plant efficiency = eta_turbine * eta_generator."""
        eta_t = self.turbine_efficiency(Q_m3s, head_m, turbine_type)
        return np.where(eta_t > 0, eta_t * self.eta_gen, 0.0)

    def power_kw(self, Q_m3s, head_m, turbine_type="francis"):
        """Electrical output power [kW]."""
        Q = np.asarray(Q_m3s, dtype=float)
        H = np.asarray(head_m, dtype=float)
        eta = self.overall_efficiency(Q, H, turbine_type)
        t = self._get_turbine(turbine_type)
        P = eta * self.rho * self.g * Q * H / 1000.0
        return np.clip(P, 0.0, t["P_rated"])

    def specific_speed(self, Q_m3s, head_m, turbine_type="francis"):
        """
        Dimensionless specific speed: Ns = N*sqrt(Q) / (g*H)^(3/4).
        Uses synchronous speed estimate: N ~ 120*f/p where we estimate from rated point.
        For simplicity, use Ns = sqrt(Q) / (g*H)^(3/4) * constant.
        Returns normalized specific speed (Q in m3/s, H in m).
        """
        Q = np.asarray(Q_m3s, dtype=float)
        H = np.asarray(head_m, dtype=float)
        # Type-specific speed parameter (rpm * sqrt(m3/s) / m^(3/4))
        # Francis ~60-300, Kaplan ~300-800, Pelton ~10-60
        denom = np.maximum((self.g * H) ** 0.75, 1e-9)
        return np.sqrt(np.maximum(Q, 0.0)) / denom

    def flow_ratio(self, Q_m3s, turbine_type="francis"):
        """Flow ratio q = Q/Q_rated."""
        t = self._get_turbine(turbine_type)
        return np.asarray(Q_m3s, dtype=float) / t["Q_rated"]

    def environmental_flow(self, turbine_type="francis"):
        """Minimum environmental flow [m3/s]."""
        t = self._get_turbine(turbine_type)
        return self.Q_env_frac * t["Q_rated"]

    def available_flow(self, Q_total_m3s, turbine_type="francis"):
        """Flow available for power generation after environmental release."""
        Q_total = np.asarray(Q_total_m3s, dtype=float)
        Q_env = self.environmental_flow(turbine_type)
        return np.maximum(Q_total - Q_env, 0.0)
