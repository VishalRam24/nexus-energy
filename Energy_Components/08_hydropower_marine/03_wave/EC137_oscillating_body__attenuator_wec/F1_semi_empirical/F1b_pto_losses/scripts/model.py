"""
EC137 — Oscillating Body / Attenuator WEC — F1b PTO Loss Model

Extends F1a with:
  1. Sea-state-dependent CWR:
     CWR(H_s, T_e) = cwr_design * (H_s/H_s_design)^cwr_H_s_exp
                   * exp(-0.5 * ((T_e - T_e_design)/sigma_T)^2)
     CWR increases slightly with wave height (more joint motion),
     and decreases off design period (reduced angular coupling).

  2. PTO efficiency vs sea state:
     eta_pto(H_s) = eta_pto_design + pto_efficiency_H_s_coeff * (H_s - H_s_design)
     Hydraulic PTO efficiency varies with wave height (stroke and force levels).
     Clamped to [0.50, 0.92].

  3. Seawater density correction:
     rho(T, S) = rho_ref + 0.8*(S - S_ref) - 0.15*(T - T_ref)

  4. Directionality factor:
     Attenuator must be aligned with wave direction.
     In multidirectional seas, effective capture is reduced.
     dir_factor ≈ 0.92 (azimuthal averaging over ±30° spread).

References:
    Henderson, R. (2006). Design, simulation, and testing of a novel hydraulic
        power take-off system for the Pelamis wave energy converter.
        Applied Ocean Research, 28(6), 297-307.
    Yemm, R. et al. (2012). Pelamis: experience from concept to connection.
        Phil. Trans. R. Soc. A, 370, 365-380.
    Babarit, A. (2015). Renew. Sustain. Energy Rev., 46, 291-306.
"""

import numpy as np

_G = 9.81


class AttenuatorWECF1b:
    """Pelamis-type attenuator WEC — sea-state-dependent CWR and PTO efficiency."""

    def __init__(self, params: dict):
        d = params["device"]
        self.length           = d["length_m"]["value"]
        self.width            = d["segment_width_m"]["value"]
        self.n_joints         = d["n_joints"]["value"]
        self.cwr_design       = d["cwr_design"]["value"]
        self.eta_pto_design   = d["eta_pto_design"]["value"]
        self.eta_elec         = d["eta_electrical"]["value"]
        self.rho_ref          = d["rho_water"]["value"]
        self.T_e_design       = d["T_e_design"]["value"]
        self.H_s_design       = d["H_s_design"]["value"]
        self.cwr_Hs_exp       = d["cwr_H_s_exponent"]["value"]
        self.sigma_T          = d["cwr_T_e_sigma"]["value"]
        self.pto_Hs_coeff     = d["pto_efficiency_H_s_coeff"]["value"]
        self.dir_factor       = d["directionality_factor"]["value"]
        self.S_ref            = d["S_ref_psu"]["value"]
        self.T_ref            = d["T_ref_C"]["value"]

    def seawater_density(self, T_C=None, S_psu=None):
        """rho(T, S) [kg/m3]."""
        T = self.T_ref if T_C is None else np.asarray(T_C, dtype=float)
        S = self.S_ref if S_psu is None else np.asarray(S_psu, dtype=float)
        return self.rho_ref + 0.8 * (S - self.S_ref) - 0.15 * (T - self.T_ref)

    def capture_width_ratio(self, H_s, T_e):
        """
        Sea-state-dependent CWR.

        CWR(H_s, T_e) = cwr_design
            * (H_s/H_s_design)^cwr_Hs_exp         [H_s dependence]
            * exp(-0.5*((T_e-T_e_design)/sigma_T)^2)  [period resonance]

        Clamped to [0, 1.0] (physical).
        """
        H_s = np.asarray(H_s, dtype=float)
        T_e = np.asarray(T_e, dtype=float)
        H_s_safe = np.maximum(H_s, 0.01)
        cwr_Hs = (H_s_safe / self.H_s_design) ** self.cwr_Hs_exp
        cwr_Te = np.exp(-0.5 * ((T_e - self.T_e_design) / self.sigma_T) ** 2)
        return np.clip(self.cwr_design * cwr_Hs * cwr_Te, 0.0, 1.0)

    def pto_efficiency(self, H_s):
        """
        Hydraulic PTO efficiency as function of significant wave height.

        eta_pto(H_s) = eta_pto_design + pto_Hs_coeff * (H_s - H_s_design)

        At design H_s: eta_pto = eta_pto_design.
        At higher H_s: PTO may be less efficient (over-stroke, relief valves active).
        At lower H_s: also less efficient (under-stroke, low pressure).
        Clamped to physical range [0.50, 0.92].
        """
        H_s = np.asarray(H_s, dtype=float)
        eta = self.eta_pto_design + self.pto_Hs_coeff * (H_s - self.H_s_design)
        return np.clip(eta, 0.50, 0.92)

    def wave_power_per_metre(self, H_s, T_e, T_C=None, S_psu=None):
        """Incident wave power per unit crest width [W/m]."""
        H_s = np.asarray(H_s, dtype=float)
        T_e = np.asarray(T_e, dtype=float)
        rho = self.seawater_density(T_C, S_psu)
        return (rho * _G**2 * H_s**2 * T_e) / (64.0 * np.pi)

    def power_kw(self, H_s, T_e, T_C=None, S_psu=None, apply_directionality=True):
        """
        Electrical power output [kW] with F1b corrections.

        P = J * width * CWR(H_s,T_e) * eta_pto(H_s) * eta_elec * dir_factor
        """
        J     = self.wave_power_per_metre(H_s, T_e, T_C, S_psu)
        cwr   = self.capture_width_ratio(H_s, T_e)
        eta_p = self.pto_efficiency(H_s)
        P_w   = J * self.width
        P_e   = P_w * cwr * eta_p * self.eta_elec
        if apply_directionality:
            P_e = P_e * self.dir_factor
        return np.clip(P_e, 0.0, None) / 1e3

    def overall_efficiency(self, H_s, T_e, apply_directionality=True):
        """Wave-to-wire efficiency."""
        cwr   = self.capture_width_ratio(H_s, T_e)
        eta_p = self.pto_efficiency(H_s)
        eff   = cwr * eta_p * self.eta_elec
        if apply_directionality:
            eff *= self.dir_factor
        return eff

    def power_per_joint_kw(self, H_s, T_e, T_C=None, S_psu=None):
        """Power per PTO joint [kW]."""
        return self.power_kw(H_s, T_e, T_C, S_psu) / self.n_joints

    def density_effect_pct(self, T_C, S_psu):
        rho = self.seawater_density(T_C, S_psu)
        return (rho - self.rho_ref) / self.rho_ref * 100.0
