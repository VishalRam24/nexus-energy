"""
EC137 — Oscillating Body / Attenuator WEC — F1a Wave Power Model

Incident wave power resource per metre crest:
    J = (rho * g^2 * H_s^2 * T_e) / (64 * pi)   [W/m]

Attenuator (Pelamis-type) device output:
    P_elec = J * W * CWR * eta_pto * eta_electrical

where:
    W   = segment_width_m  (effective width normal to wave propagation)
    CWR = capture width ratio (0.20-0.35 for attenuators aligned with waves)
    eta_pto        ~ 0.75-0.85 (hydraulic + generator)
    eta_electrical ~ 0.92-0.96

Note: CWR accounts for the fact that attenuators extract energy along their
length from multiple joints. The parameter already integrates the joint count.

References:
    Henderson (2006). Applied Ocean Research, 28, 297-307.
    Yemm et al. (2012). Phil. Trans. R. Soc. A, 370, 365-380. (Pelamis P2)
    Babarit (2015). Renew. Sustain. Energy Rev., 46, 291-306.
"""

import numpy as np

_G = 9.81


class AttenuatorWECF1a:
    """Pelamis-type oscillating body attenuator WEC."""

    def __init__(self, params: dict):
        d = params["device"]
        self.length         = d["length_m"]["value"]
        self.width          = d["segment_width_m"]["value"]
        self.n_joints       = d["n_joints"]["value"]
        self.cwr            = d["cwr"]["value"]
        self.eta_pto        = d["eta_pto"]["value"]
        self.eta_elec       = d["eta_electrical"]["value"]
        self.rho            = d["rho_water"]["value"]

    def wave_power_per_metre(self, H_s, T_e):
        """Incident wave power per unit crest width [W/m]."""
        H_s = np.asarray(H_s, dtype=float)
        T_e = np.asarray(T_e, dtype=float)
        return (self.rho * _G**2 * H_s**2 * T_e) / (64.0 * np.pi)

    def power_kw(self, H_s, T_e, cwr=None):
        """
        Electrical power output [kW].

        Parameters
        ----------
        H_s : significant wave height [m]
        T_e : energy period [s]
        cwr : capture width ratio override
        """
        if cwr is None:
            cwr = self.cwr
        cwr = np.asarray(cwr, dtype=float)
        J   = self.wave_power_per_metre(H_s, T_e)         # W/m
        P_w = J * self.width                               # W incident on device width
        P_e = P_w * cwr * self.eta_pto * self.eta_elec    # W electrical
        return np.clip(P_e, 0.0, None) / 1e3              # kW

    def overall_efficiency(self, cwr=None):
        """Wave-to-wire efficiency."""
        if cwr is None:
            cwr = self.cwr
        return float(cwr) * self.eta_pto * self.eta_elec

    def rated_power_density_kw_per_m(self, H_s, T_e):
        """Power per unit device length [kW/m]."""
        return self.power_kw(H_s, T_e) / self.length
