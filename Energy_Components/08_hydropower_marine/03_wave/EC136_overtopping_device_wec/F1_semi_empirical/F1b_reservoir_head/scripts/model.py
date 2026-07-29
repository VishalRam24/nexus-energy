"""
EC136 — Overtopping Device WEC — F1b Reservoir Head / Variable Turbine Efficiency

Extends F1a with:
  1. Variable reservoir head:
     Reservoir head varies with overtopping flow; modelled as:
     h(H_s) = h_design * (H_s/H_s_design)^alpha  [fill rate increases with waves]
     where alpha = overtopping_Hs_exp (typically 1.5).

  2. Turbine efficiency vs head:
     eta_turbine(h) = eta_peak * (1 - k_h * (h/h_design - 1)^2)
     Low-head Kaplan turbine efficiency drops at off-design head.

  3. Seawater density correction:
     rho(T, S) = rho_ref + 0.8*(S - S_ref) - 0.15*(T - T_ref)
     Affects wave power resource and turbine power.

  4. Two power calculation pathways:
     (a) Wave resource path: P = J * width * eta_ramp * eta_turbine(h) * eta_gen
     (b) Reservoir flow path: P = rho*g*h*Q * eta_turbine(h) * eta_gen
     Both are provided for cross-validation.

References:
    Kofoed, J.P. et al. (2006). Prototype testing of the wave energy converter
        Wave Dragon. Coastal Engineering, 53, 859-867.
    Margheritini, L., Vicinanza, D. & Frigaard, P. (2009). Renew. Energy, 34, 1473-1479.
    Wave Dragon AS (2010). Wave Dragon — Offshore WEC Technical Report.
"""

import numpy as np

_G = 9.81


class OvertoppingWECF1b:
    """Overtopping WEC — variable reservoir head, turbine efficiency vs head, density."""

    def __init__(self, params: dict):
        d = params["device"]
        self.width             = d["ramp_width_m"]["value"]
        self.h_design          = d["reservoir_height_design_m"]["value"]
        self.q_design          = d["overtopping_rate_design"]["value"]   # m3/s/m
        self.eta_turb_peak     = d["eta_turbine_peak"]["value"]
        self.eta_generator     = d["eta_generator"]["value"]
        self.eta_ramp          = d["eta_ramp_design"]["value"]
        self.rho_ref           = d["rho_water"]["value"]
        self.k_h               = d["k_head_turbine"]["value"]
        self.h_min             = d["h_min_turbine"]["value"]
        self.Hs_exp            = d["overtopping_Hs_exp"]["value"]
        self.S_ref             = d["S_ref_psu"]["value"]
        self.T_ref             = d["T_ref_C"]["value"]

    def seawater_density(self, T_C=None, S_psu=None):
        """rho(T, S) [kg/m3]."""
        T = self.T_ref if T_C is None else np.asarray(T_C, dtype=float)
        S = self.S_ref if S_psu is None else np.asarray(S_psu, dtype=float)
        return self.rho_ref + 0.8 * (S - self.S_ref) - 0.15 * (T - self.T_ref)

    def reservoir_head_m(self, H_s):
        """
        Effective reservoir head as function of significant wave height.

        h(H_s) = h_design * (H_s/H_s_design)^alpha

        At higher waves: more overtopping → higher mean reservoir level.
        This is an approximation of the quasi-steady reservoir dynamics.
        """
        H_s      = np.asarray(H_s, dtype=float)
        H_s_ref  = np.sqrt(self.h_design / self.q_design)  # rough design point estimate
        h_s_ref  = max(H_s_ref, 0.5)                       # minimum reference point
        h        = self.h_design * (H_s / h_s_ref) ** self.Hs_exp
        return np.clip(h, 0.0, 3.0 * self.h_design)       # cap at 3x design

    def turbine_efficiency(self, h):
        """
        Low-head Kaplan turbine efficiency vs reservoir head.
        eta(h) = eta_peak * (1 - k_h * (h/h_design - 1)^2)
        Below h_min: turbine cannot operate.
        """
        h = np.asarray(h, dtype=float)
        h_ratio = h / self.h_design
        eta     = self.eta_turb_peak * (1.0 - self.k_h * (h_ratio - 1.0) ** 2)
        eta     = np.where(h < self.h_min, 0.0, eta)
        return np.clip(eta, 0.0, self.eta_turb_peak)

    def wave_power_per_metre(self, H_s, T_e, T_C=None, S_psu=None):
        """Incident wave power per unit crest width [W/m]."""
        H_s = np.asarray(H_s, dtype=float)
        T_e = np.asarray(T_e, dtype=float)
        rho = self.seawater_density(T_C, S_psu)
        return (rho * _G**2 * H_s**2 * T_e) / (64.0 * np.pi)

    def power_kw(self, H_s, T_e, T_C=None, S_psu=None):
        """
        Electrical power output [kW] via wave resource pathway.
        P = J * width * eta_ramp * eta_turbine(h(H_s)) * eta_generator
        """
        J     = self.wave_power_per_metre(H_s, T_e, T_C, S_psu)
        P_w   = J * self.width
        h     = self.reservoir_head_m(H_s)
        eta_t = self.turbine_efficiency(h)
        P_e   = P_w * self.eta_ramp * eta_t * self.eta_generator
        return np.clip(P_e, 0.0, None) / 1e3

    def power_from_flow_kw(self, H_s, T_C=None, S_psu=None):
        """
        Electrical power from overtopping flow pathway.
        P = rho * g * h * Q * eta_turbine(h) * eta_gen
        where Q = q_design * width * (H_s/h_s_ref)^Hs_exp
        """
        H_s   = np.asarray(H_s, dtype=float)
        rho   = self.seawater_density(T_C, S_psu)
        h     = self.reservoir_head_m(H_s)
        eta_t = self.turbine_efficiency(h)
        h_s_ref = max(np.sqrt(self.h_design / self.q_design), 0.5)
        Q     = self.q_design * self.width * (H_s / h_s_ref) ** self.Hs_exp
        Q     = np.clip(Q, 0.0, None)
        P_e   = rho * _G * h * Q * eta_t * self.eta_generator
        return np.clip(P_e, 0.0, None) / 1e3

    def overall_efficiency(self, H_s):
        """Wave-to-wire efficiency = eta_ramp * eta_turbine(h) * eta_gen."""
        h     = self.reservoir_head_m(H_s)
        eta_t = self.turbine_efficiency(h)
        return self.eta_ramp * eta_t * self.eta_generator

    def density_effect_pct(self, T_C, S_psu):
        rho = self.seawater_density(T_C, S_psu)
        return (rho - self.rho_ref) / self.rho_ref * 100.0
