"""
EC135 — Point Absorber WEC — F1b Resonance / PTO Model

Extends F1a with:
  1. PTO efficiency as function of power level:
     eta_pto(P) = eta_pto_rated - pto_partload_coeff * (1 - P/P_rated)
     Linear generator efficiency drops at part load.
     Clamped to [eta_pto_min, eta_pto_rated].

  2. Seawater density correction:
     rho(T, S) = rho_ref + 0.8*(S - S_ref) - 0.15*(T - T_ref)
     More wave power in cold, saline water.

  3. Power rating cap:
     Output limited to P_rated (PTO hardware limit).

  4. Storm cutout / cut-in:
     H_s > H_s_cutout: device parks, output = 0
     H_s < H_s_cutin:  insufficient power, output = 0

  5. Resonance-dependent CWR (same Gaussian as F1a):
     CWR(T_e) = CWR_peak * exp(-0.5 * ((T_e - T_n)/sigma)^2)

References:
    Falnes, J. (2002). Ocean Waves and Oscillating Systems. Cambridge UP.
    Babarit, A. et al. (2012). Renew. Energy, 41, 44-63.
    Eriksson, M., Isberg, J. & Leijon, M. (2005). Renew. Sustain. Energy Rev., 9, 435-444.
    CorPower Ocean (2020). C4 WEC Technical Datasheet.
"""

import numpy as np

_G = 9.81


class PointAbsorberF1b:
    """Heaving point absorber WEC with PTO efficiency model and rated power limit."""

    def __init__(self, params: dict):
        d = params["device"]
        self.diameter        = d["diameter_m"]["value"]
        self.T_n             = d["T_n_s"]["value"]
        self.cwr_peak        = d["cwr_peak"]["value"]
        self.sigma           = d["bandwidth_s"]["value"]
        self.eta_pto_rated   = d["eta_pto_rated"]["value"]
        self.eta_pto_min     = 0.40  # physical lower bound
        self.eta_elec        = d["eta_electrical"]["value"]
        self.rho_ref         = d["rho_water"]["value"]
        self.P_rated_kw      = d["P_rated_kw"]["value"]
        self.H_s_cutout      = d["H_s_cutout"]["value"]
        self.H_s_cutin       = d["H_s_cutin"]["value"]
        self.pto_plc         = d["pto_partload_coeff"]["value"]
        self.S_ref           = d["S_ref_psu"]["value"]
        self.T_ref           = d["T_ref_C"]["value"]

    def seawater_density(self, T_C=None, S_psu=None):
        """rho(T, S) [kg/m3]."""
        T = self.T_ref if T_C is None else np.asarray(T_C, dtype=float)
        S = self.S_ref if S_psu is None else np.asarray(S_psu, dtype=float)
        return self.rho_ref + 0.8 * (S - self.S_ref) - 0.15 * (T - self.T_ref)

    def wave_power_per_metre(self, H_s, T_e, T_C=None, S_psu=None):
        """Incident wave power per unit crest width [W/m] with density correction."""
        H_s = np.asarray(H_s, dtype=float)
        T_e = np.asarray(T_e, dtype=float)
        rho = self.seawater_density(T_C, S_psu)
        return (rho * _G**2 * H_s**2 * T_e) / (64.0 * np.pi)

    def capture_width_ratio(self, T_e):
        """Resonance Gaussian CWR profile."""
        T_e = np.asarray(T_e, dtype=float)
        return self.cwr_peak * np.exp(-0.5 * ((T_e - self.T_n) / self.sigma) ** 2)

    def pto_efficiency(self, P_raw_kw):
        """
        PTO efficiency as function of mechanical power level.

        eta_pto(P) = eta_pto_rated - pto_partload_coeff * (1 - P/P_rated)
        Linear generator: efficiency drops at part-load (smaller flux linkage).
        Clamped to physical range.
        """
        P_raw_kw = np.asarray(P_raw_kw, dtype=float)
        power_fraction = np.clip(P_raw_kw / self.P_rated_kw, 0.0, 1.0)
        eta = self.eta_pto_rated - self.pto_plc * (1.0 - power_fraction)
        return np.clip(eta, self.eta_pto_min, self.eta_pto_rated)

    def power_kw(self, H_s, T_e, T_C=None, S_psu=None):
        """
        Electrical power output [kW] with all F1b corrections.

        Steps:
          1. Compute wave power density J(rho)
          2. Apply resonance CWR(T_e)
          3. Compute mechanical power (no PTO losses yet)
          4. Apply PTO efficiency (function of mechanical power fraction)
          5. Apply electrical efficiency
          6. Apply rated power cap
          7. Apply cut-in / cut-out
        """
        H_s = np.asarray(H_s, dtype=float)
        T_e = np.asarray(T_e, dtype=float)

        J    = self.wave_power_per_metre(H_s, T_e, T_C, S_psu)
        cwr  = self.capture_width_ratio(T_e)
        P_m  = J * self.diameter * cwr / 1e3   # kW mechanical

        eta_pto = self.pto_efficiency(P_m)
        P_e = P_m * eta_pto * self.eta_elec     # kW electrical

        # Rated power cap
        P_e = np.minimum(P_e, self.P_rated_kw)

        # Cut-in / cut-out
        P_e = np.where(H_s < self.H_s_cutin,  0.0, P_e)
        P_e = np.where(H_s >= self.H_s_cutout, 0.0, P_e)

        return np.clip(P_e, 0.0, self.P_rated_kw)

    def overall_efficiency(self, H_s, T_e, T_C=None, S_psu=None):
        """Wave-to-wire efficiency."""
        J   = self.wave_power_per_metre(H_s, T_e, T_C, S_psu)
        cwr = self.capture_width_ratio(T_e)
        P_m = J * self.diameter * cwr / 1e3
        eta_pto = self.pto_efficiency(P_m)
        return cwr * eta_pto * self.eta_elec

    def density_effect_pct(self, T_C, S_psu):
        """Power change [%] from density vs reference."""
        rho = self.seawater_density(T_C, S_psu)
        return (rho - self.rho_ref) / self.rho_ref * 100.0
