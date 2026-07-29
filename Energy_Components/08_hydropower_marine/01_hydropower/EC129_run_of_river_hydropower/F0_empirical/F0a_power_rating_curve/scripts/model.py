"""EC129 — Run-of-River Hydropower — F0a empirical power-rating curve.

Simplest fidelity: a tabulated overall-efficiency curve vs flow ratio q=Q/Q_design,
interpolated with numpy.interp. Hydraulic power P = eta(q)*rho*g*Q*H_net/1000 (kW).
Low-head Kaplan/Francis run-of-river plant.
Data source: Penche (1998); IEC 60041; reuses EC129 F1a rated parameters.
NumPy only.
"""
import numpy as np


class HydroF0a:
    def __init__(self, params):
        r = params["rated"]
        c = params["efficiency_curve"]
        self.H_design = r["H_design"]["value"]
        self.Q_design = r["Q_design"]["value"]
        self.P_rated = r["P_rated"]["value"]
        self.rho = r["rho"]["value"]
        self.g = r["g"]["value"]
        self.q_tab = np.asarray(c["q_ratio"]["value"], dtype=float)
        self.eta_tab = np.asarray(c["eta_overall"]["value"], dtype=float)

    def overall_efficiency(self, Q):
        """Interpolated overall efficiency at flow Q (clamped to table ends)."""
        q = np.asarray(Q, dtype=float) / self.Q_design
        return np.interp(q, self.q_tab, self.eta_tab)

    def power_kw(self, Q, H):
        Q = np.asarray(Q, dtype=float)
        H = np.asarray(H, dtype=float)
        eta = self.overall_efficiency(Q)
        return eta * self.rho * self.g * Q * H / 1000.0

    def capacity_factor(self, Q, H):
        return self.power_kw(Q, H) / self.P_rated
