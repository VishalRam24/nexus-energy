"""
EC134 — Oscillating Water Column (OWC) — F1b PTO Loss Model

Extends F1a with:
  1. Wells turbine efficiency vs wave period:
     eta_turbine(T_e) = eta_peak - (eta_peak - eta_min) * |T_e - T_e_design|^2 / bandwidth^2
     Clamped to [eta_min, eta_peak].
     At design period: peak efficiency. At off-design periods: efficiency degrades.

  2. Seawater density correction:
     rho(T, S) = rho_ref + 0.8*(S - S_ref) - 0.15*(T - T_ref)
     More energy in cold, saline water.

  3. Directional spreading correction:
     Real ocean waves are directionally spread → less power captured than
     from unidirectional wave model. Factor applied as multiplicative correction.

  4. Wave power formula (same as F1a):
     J = (rho * g^2 * H_s^2 * T_e) / (64 * pi)   [W/m]

References:
    Falnes, J. (2002). Ocean Waves and Oscillating Systems. Cambridge UP.
    Folley, M. ed. (2016). Numerical Modelling of Wave Energy Converters.
        Academic Press.
    EMEC (2019). Assessment of Wave Energy Resource, TR-001.
"""

import numpy as np

_G = 9.81


class OWCF1b:
    """OWC WEC — wave-to-wire model with turbine efficiency vs period and density correction."""

    def __init__(self, params: dict):
        d = params["device"]
        self.width           = d["width_m"]["value"]
        self.cwr             = d["capture_width_ratio"]["value"]
        self.eta_turb_peak   = d["eta_turbine_peak"]["value"]
        self.eta_generator   = d["eta_generator"]["value"]
        self.rho_ref         = d["rho_water"]["value"]
        self.T_e_design      = d["T_e_design"]["value"]
        self.bandwidth       = d["turbine_bandwidth_s"]["value"]
        self.eta_turb_min    = d["eta_turbine_min"]["value"]
        self.dir_factor      = d["directional_spread_factor"]["value"]
        self.S_ref           = d["S_ref_psu"]["value"]
        self.T_ref           = d["T_ref_C"]["value"]

    def seawater_density(self, T_C=None, S_psu=None):
        """rho(T, S) [kg/m3]."""
        T = self.T_ref if T_C is None else np.asarray(T_C, dtype=float)
        S = self.S_ref if S_psu is None else np.asarray(S_psu, dtype=float)
        return self.rho_ref + 0.8 * (S - self.S_ref) - 0.15 * (T - self.T_ref)

    def turbine_efficiency(self, T_e):
        """
        Wells turbine efficiency vs wave energy period.

        eta(T_e) = eta_peak - (eta_peak - eta_min) * ((T_e - T_e_design)/bandwidth)^2
        Clamped to [eta_min, eta_peak].

        At design period: eta = eta_peak.
        Away from design: efficiency drops toward eta_min.
        """
        T_e = np.asarray(T_e, dtype=float)
        deviation = (T_e - self.T_e_design) / self.bandwidth
        eta = self.eta_turb_peak - (self.eta_turb_peak - self.eta_turb_min) * deviation ** 2
        return np.clip(eta, self.eta_turb_min, self.eta_turb_peak)

    def wave_power_per_metre(self, H_s, T_e, T_C=None, S_psu=None):
        """
        Incident wave power per unit crest width [W/m] with density correction.
        J = (rho * g^2 * H_s^2 * T_e) / (64 * pi)
        """
        H_s = np.asarray(H_s, dtype=float)
        T_e = np.asarray(T_e, dtype=float)
        rho = self.seawater_density(T_C, S_psu)
        return (rho * _G**2 * H_s**2 * T_e) / (64.0 * np.pi)

    def power_kw(self, H_s, T_e, T_C=None, S_psu=None, apply_directionality=True):
        """
        Electrical power output [kW] with all F1b corrections.

        P_e = J * width * CWR * eta_turbine(T_e) * eta_generator * dir_factor
        """
        J     = self.wave_power_per_metre(H_s, T_e, T_C, S_psu)
        eta_t = self.turbine_efficiency(T_e)
        P_w   = J * self.width
        P_e   = P_w * self.cwr * eta_t * self.eta_generator
        if apply_directionality:
            P_e = P_e * self.dir_factor
        return np.clip(P_e, 0.0, None) / 1e3

    def overall_efficiency(self, T_e, apply_directionality=True):
        """Wave-to-wire efficiency = CWR * eta_turbine(T_e) * eta_gen * dir_factor."""
        eta_t = self.turbine_efficiency(T_e)
        eff   = self.cwr * eta_t * self.eta_generator
        if apply_directionality:
            eff *= self.dir_factor
        return eff

    def density_effect_pct(self, T_C, S_psu):
        """Power change [%] relative to reference density."""
        rho = self.seawater_density(T_C, S_psu)
        return (rho - self.rho_ref) / self.rho_ref * 100.0
