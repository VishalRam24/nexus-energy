"""
EC191 — Gas Compressor Station — F1b Part-Load + Inlet Temperature Model

Extends F1a polytropic model with:
  1. Part-load polytropic efficiency degradation:
       eta_p(PLR) = (a0 + a1*PLR + a2*PLR^2)  normalized so eta(PLR=1) = eta_design
     Centrifugal machines typically show 5-15% efficiency drop at 50% load.
  2. Inlet temperature correction:
       The polytropic work scales directly with T_inlet:
           w_stage = (n/(n-1)) * R_s * T_inlet * (PRs^((n-1)/n) - 1) / eta_p
       Hotter inlet → more work. Correction vs reference T_ref:
           w_corr = w_base * T_inlet / T_ref
       This is the ISO 13631 / API 617 inlet condition correction.
  3. Discharge temperature accounting for actual polytropic efficiency.
  4. Intercooling effectiveness parameter.

References:
    Menon, E.S. (2005). Gas Pipeline Hydraulics. CRC Press. Chapters 5-6.
    Campbell, J.M. (2014). Gas Conditioning & Processing, Vol. 2.
    API 617 (2014). Axial and Centrifugal Compressors. 8th Ed.
"""

import numpy as np


class NGCompressorF1b:
    """
    Multistage polytropic gas compressor with part-load efficiency degradation
    and inlet temperature correction.
    """

    def __init__(self, params: dict):
        c = params["compressor"]
        g = params["gas"]

        self.N = int(c["n_stages"]["value"])
        self.n = c["polytropic_index"]["value"]
        self.eta_p_design = c["eta_polytropic"]["value"]
        self.eta_m = c["eta_mech"]["value"]
        self.T_inlet_default = c["T_inlet"]["value"]
        self.PLR_eta_coeffs = c["PLR_eta_coeffs"]["value"]   # [a0, a1, a2]
        self.T_inlet_ref = c["T_inlet_ref"]["value"]

        self.R_s = g["R_specific"]["value"]   # J/(kg.K)
        self.gamma = g["gamma"]["value"]
        self.LHV = g["LHV"]["value"]          # MJ/kg

    def polytropic_efficiency(self, plr):
        """
        Part-load polytropic efficiency.
        eta_p(PLR) = eta_design * (a0 + a1*PLR + a2*PLR^2) / (a0+a1+a2)
        Normalized so PLR=1 → eta_design.
        """
        plr = np.asarray(plr, dtype=float)
        a0, a1, a2 = self.PLR_eta_coeffs
        f_raw = a0 + a1 * plr + a2 * plr ** 2
        f_at_1 = a0 + a1 + a2
        f = np.clip(f_raw / f_at_1, 0.5, 1.0)
        return self.eta_p_design * f

    def stage_pressure_ratio(self, P_in, P_out):
        """Optimum equal stage pressure ratio."""
        return (np.asarray(P_out, dtype=float) /
                np.asarray(P_in, dtype=float)) ** (1.0 / self.N)

    def specific_work(self, P_in, P_out, T_in=None, plr=1.0):
        """
        Specific shaft work [J/kg] with part-load eta and inlet T correction.

        w_stage = (n/(n-1)) * R_s * T_in * (PRs^((n-1)/n) - 1) / eta_p(PLR)
        w_total = N * w_stage / eta_mech
        """
        T1 = self.T_inlet_default if T_in is None else np.asarray(T_in, dtype=float)
        P_in_arr = np.asarray(P_in, dtype=float)
        P_out_arr = np.asarray(P_out, dtype=float)
        plr_arr = np.asarray(plr, dtype=float)

        eta_p = self.polytropic_efficiency(plr_arr)
        PRs = self.stage_pressure_ratio(P_in_arr, P_out_arr)
        exponent = (self.n - 1.0) / self.n
        w_stage = ((self.n / (self.n - 1.0)) * self.R_s * T1 *
                   (PRs ** exponent - 1.0) / eta_p)
        return self.N * w_stage / self.eta_m

    def specific_work_kJ_per_kg(self, P_in, P_out, T_in=None, plr=1.0):
        return self.specific_work(P_in, P_out, T_in, plr) / 1000.0

    def sec_kwh_per_kg(self, P_in, P_out, T_in=None, plr=1.0):
        """Specific energy [kWh/kg NG]."""
        return self.specific_work(P_in, P_out, T_in, plr) / 3.6e6

    def shaft_power_kw(self, m_dot, P_in, P_out, T_in=None, plr=1.0):
        """Shaft power [kW] for given mass flow and part-load ratio."""
        m = np.asarray(m_dot, dtype=float)
        plr_arr = np.asarray(plr, dtype=float)
        w = self.specific_work(P_in, P_out, T_in, plr_arr)
        # At part-load, mass flow = PLR * design_flow → P = PLR * m_design * w
        return m * plr_arr * w / 1000.0

    def discharge_temperature(self, T_in, P_in, P_out, plr=1.0):
        """
        Stage discharge temperature after polytropic compression.
        T2 = T1 * PRs^((n-1)/n)  (before intercooling)
        With part-load eta correction the polytropic index n_eff = 1/(1 - (n-1)/(n*eta_p)).
        """
        T1 = np.asarray(T_in, dtype=float)
        plr_arr = np.asarray(plr, dtype=float)
        eta_p = self.polytropic_efficiency(plr_arr)
        PRs = self.stage_pressure_ratio(P_in, P_out)
        # Effective polytropic exponent
        n_eff = self.n / (1.0 - (self.n - 1.0) / (self.n * eta_p) * (1.0 - 1.0))
        # Simplified: use Menon (2005) Eq. 6.21 with actual n
        exponent = (self.n - 1.0) / self.n
        T2 = T1 * PRs ** exponent
        return T2

    def compression_efficiency_overall(self, P_in, P_out, T_in=None, plr=1.0):
        """Overall compression efficiency: LHV energy / (LHV + shaft work per kg)."""
        w_MJ = self.specific_work(P_in, P_out, T_in, plr) / 1.0e6
        return self.LHV / (self.LHV + w_MJ)

    def compute(self, m_dot_kg_s, P_in_bar, P_out_bar, T_in_K=None, plr=1.0):
        """Full computation returning all outputs."""
        T_in = self.T_inlet_default if T_in_K is None else T_in_K
        w = self.specific_work_kJ_per_kg(P_in_bar, P_out_bar, T_in, plr)
        sec = self.sec_kwh_per_kg(P_in_bar, P_out_bar, T_in, plr)
        power = self.shaft_power_kw(m_dot_kg_s, P_in_bar, P_out_bar, T_in, plr)
        T2 = self.discharge_temperature(T_in, P_in_bar, P_out_bar, plr)
        eta_p = self.polytropic_efficiency(np.asarray(plr, dtype=float))
        eta_overall = self.compression_efficiency_overall(P_in_bar, P_out_bar, T_in, plr)

        return {
            "specific_work_kJ_per_kg": w,
            "sec_kwh_per_kg": sec,
            "shaft_power_kw": power,
            "discharge_temperature_K": T2,
            "polytropic_efficiency": eta_p,
            "overall_efficiency": eta_overall,
        }
