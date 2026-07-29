"""
EC136 — Overtopping WEC — F1a Wave Power Model

Incident wave power resource per metre crest:
    J = (rho * g^2 * H_s^2 * T_e) / (64 * pi)   [W/m]

Overtopping device output:
    P_elec = J * W * eta_ramp * eta_turbine * eta_generator

Alternatively, if overtopping flow Q is known:
    P_elec = rho * g * h * Q * eta_turbine * eta_generator

where:
    W = ramp width [m]
    h = reservoir head [m]
    Q = overtopping flow rate [m^3/s]
    eta_ramp  ~ 0.15-0.25 (fraction of incident wave energy captured)
    eta_turbine ~ 0.75-0.85 (Kaplan low-head turbine)

References:
    Kofoed et al. (2006). Coastal Engineering, 53, 859-867.
    Margheritini, Vicinanza & Frigaard (2009). Renew. Energy, 34, 1473-1479.
    Wave Dragon AS (2010). Wave Dragon — Offshore WEC Technical Report.
"""

import numpy as np

_G = 9.81


class OvertoppingWECF1a:
    """Overtopping WEC — ramp + reservoir + low-head turbine."""

    def __init__(self, params: dict):
        d = params["device"]
        self.width         = d["ramp_width_m"]["value"]
        self.head          = d["reservoir_height_m"]["value"]
        self.q_per_m       = d["overtopping_rate_m3_per_s_per_m"]["value"]
        self.eta_turbine   = d["eta_turbine"]["value"]
        self.eta_generator = d["eta_generator"]["value"]
        self.eta_ramp      = d["eta_ramp"]["value"]
        self.rho           = d["rho_water"]["value"]

    def wave_power_per_metre(self, H_s, T_e):
        """Incident wave power per unit crest width [W/m]."""
        H_s = np.asarray(H_s, dtype=float)
        T_e = np.asarray(T_e, dtype=float)
        return (self.rho * _G**2 * H_s**2 * T_e) / (64.0 * np.pi)

    def power_kw(self, H_s, T_e):
        """
        Electrical power output [kW].

        Energy pathway: wave → ramp overtopping → potential energy in reservoir
        → low-head turbine → generator.
        """
        J = self.wave_power_per_metre(H_s, T_e)  # W/m
        P_w = J * self.width                       # W incident on full device
        P_e = P_w * self.eta_ramp * self.eta_turbine * self.eta_generator
        return np.clip(P_e, 0.0, None) / 1e3      # kW

    def power_from_flow_kw(self, Q_m3s, head=None):
        """
        Power from overtopping flow directly.

        P = rho * g * h * Q * eta_turbine * eta_generator
        """
        if head is None:
            head = self.head
        Q = np.asarray(Q_m3s, dtype=float)
        P_e = self.rho * _G * head * Q * self.eta_turbine * self.eta_generator
        return np.clip(P_e, 0.0, None) / 1e3  # kW

    def overall_efficiency(self):
        """Wave-to-wire efficiency = eta_ramp * eta_turbine * eta_generator."""
        return self.eta_ramp * self.eta_turbine * self.eta_generator
