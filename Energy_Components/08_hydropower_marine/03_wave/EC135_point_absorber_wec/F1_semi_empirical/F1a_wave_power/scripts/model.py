"""
EC135 — Point Absorber WEC — F1a Wave Power Model

Wave resource:
    J = (rho * g^2 * H_s^2 * T_e) / (64 * pi)   [W/m]

Resonance-dependent capture width ratio (Gaussian around T_n):
    CWR(T_e) = CWR_peak * exp(-0.5 * ((T_e - T_n) / sigma)^2)

Electrical output:
    P_elec = J * D * CWR(T_e) * eta_pto * eta_elec

where D = buoy diameter (point absorber acts on its projected width).

References:
    Falnes (2002). Ocean Waves and Oscillating Systems. Cambridge UP.
    Babarit et al. (2012). Renew. Energy, 41, 44-63.
    Eriksson, Isberg & Leijon (2005). Renew. Sustain. Energy Rev., 9, 435-444.
"""

import numpy as np

_G = 9.81


class PointAbsorberF1a:
    """Heaving point absorber WEC with resonance-dependent CWR."""

    def __init__(self, params: dict):
        d = params["device"]
        self.diameter    = d["diameter_m"]["value"]       # m
        self.T_n         = d["T_n_s"]["value"]            # resonance period [s]
        self.cwr_peak    = d["cwr_peak"]["value"]         # peak CWR at resonance
        self.sigma       = d["bandwidth_s"]["value"]      # Gaussian sigma [s]
        self.eta_pto     = d["eta_pto"]["value"]
        self.eta_elec    = d["eta_electrical"]["value"]
        self.rho         = d["rho_water"]["value"]

    def wave_power_per_metre(self, H_s, T_e):
        """Incident wave power per unit crest width [W/m]."""
        H_s = np.asarray(H_s, dtype=float)
        T_e = np.asarray(T_e, dtype=float)
        return (self.rho * _G**2 * H_s**2 * T_e) / (64.0 * np.pi)

    def capture_width_ratio(self, T_e):
        """Resonance Gaussian CWR profile."""
        T_e = np.asarray(T_e, dtype=float)
        return self.cwr_peak * np.exp(-0.5 * ((T_e - self.T_n) / self.sigma)**2)

    def power_kw(self, H_s, T_e):
        """
        Electrical power output [kW].

        Parameters
        ----------
        H_s : significant wave height [m]
        T_e : energy period [s]
        """
        J   = self.wave_power_per_metre(H_s, T_e)        # W/m
        cwr = self.capture_width_ratio(T_e)               # dimensionless
        P_w = J * self.diameter                            # W incident on device
        P_e = P_w * cwr * self.eta_pto * self.eta_elec   # W electrical
        return np.clip(P_e, 0.0, None) / 1e3             # kW

    def overall_efficiency(self, T_e):
        """Wave-to-wire efficiency at given T_e."""
        return self.capture_width_ratio(T_e) * self.eta_pto * self.eta_elec
