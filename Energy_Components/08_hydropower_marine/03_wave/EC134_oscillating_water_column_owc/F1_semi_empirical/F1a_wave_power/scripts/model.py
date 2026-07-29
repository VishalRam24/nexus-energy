"""
EC134 — Oscillating Water Column (OWC) — F1a Wave Power Model

Wave power resource:
    P_wave = (rho * g^2 * H_s^2 * T_e) / (64 * pi)   [W/m of wave crest]

Device electrical output:
    P_elec = P_wave * width * CWR * eta_turbine * eta_generator

where:
    H_s   = significant wave height [m]
    T_e   = energy period [s]
    CWR   = capture width ratio (0.1 – 0.3 for OWC)
    Wells turbine eta ~ 0.5 – 0.7

References:
    Falnes (2002). Ocean Waves and Oscillating Systems. Cambridge UP.
    Folley (2016). Numerical Modelling of Wave Energy Converters. Academic Press.
    EMEC (2019). Assessment of Wave Energy Resource, TR-001.
"""

import numpy as np

_G = 9.81  # m/s^2


class OWCF1a:
    """OWC WEC — semi-empirical wave-to-wire power model."""

    def __init__(self, params: dict):
        d = params["device"]
        self.width          = d["width_m"]["value"]          # m
        self.cwr            = d["capture_width_ratio"]["value"]
        self.eta_turbine    = d["eta_turbine"]["value"]
        self.eta_generator  = d["eta_generator"]["value"]
        self.rho            = d["rho_water"]["value"]        # kg/m3

    # ------------------------------------------------------------------
    # Resource
    # ------------------------------------------------------------------
    def wave_power_per_metre(self, H_s, T_e):
        """
        Incident wave power per unit crest width [W/m].

        P_wave = (rho * g^2 * H_s^2 * T_e) / (64 * pi)
        """
        H_s = np.asarray(H_s, dtype=float)
        T_e = np.asarray(T_e, dtype=float)
        return (self.rho * _G**2 * H_s**2 * T_e) / (64.0 * np.pi)

    # ------------------------------------------------------------------
    # Device output
    # ------------------------------------------------------------------
    def power_kw(self, H_s, T_e, cwr=None):
        """
        Electrical power output [kW].

        Parameters
        ----------
        H_s : significant wave height [m]
        T_e : energy period [s]
        cwr : capture width ratio override (default from params)
        """
        if cwr is None:
            cwr = self.cwr
        cwr = np.asarray(cwr, dtype=float)
        J = self.wave_power_per_metre(H_s, T_e)  # W/m
        P_w = J * self.width                      # W (total incident on device)
        P_e = P_w * cwr * self.eta_turbine * self.eta_generator  # W
        return np.clip(P_e, 0.0, None) / 1e3     # kW

    def capture_width_m(self, H_s, T_e, cwr=None):
        """Equivalent capture width [m] = CWR * device_width."""
        if cwr is None:
            cwr = self.cwr
        return float(cwr) * self.width

    def overall_efficiency(self, cwr=None):
        """Wave-to-wire efficiency = CWR * eta_turbine * eta_generator."""
        if cwr is None:
            cwr = self.cwr
        return float(cwr) * self.eta_turbine * self.eta_generator
