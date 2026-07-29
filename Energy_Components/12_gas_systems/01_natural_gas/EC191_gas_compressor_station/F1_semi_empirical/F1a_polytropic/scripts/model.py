"""
EC191 — Gas Compressor Station — F1a Multistage Polytropic Model

Identical thermodynamic framework to EC016 (H2 compressor) but for natural gas
(γ=1.31, M=16.04 g/mol, R_s=518.3 J/(kg·K)), with typical station compression
ratios of 1.3–1.5 per stage.

Specific polytropic work per stage (intercooled to T1):
    w_stage = (n / (n-1)) * R_s * T1 * (PR_s^((n-1)/n) - 1) / eta_p   [J/kg]

Optimum equal stage pressure ratio:
    PR_stage = (P_out / P_in)^(1/N)

Total specific shaft work:
    w_total = N * w_stage / eta_mech    [J/kg]

Stage discharge temperature (before intercooling):
    T2 = T1 * PR_stage^((n-1)/n)       [K]

References:
    Menon, E.S. (2005). Gas Pipeline Hydraulics. CRC Press. Chapters 5-6.
    Campbell, J.M. (2014). Gas Conditioning & Processing, Vol. 2. Campbell Petroleum Series.
"""

import numpy as np


class NGCompressorF1a:
    """Multistage polytropic natural gas compressor station with intercooling."""

    def __init__(self, params: dict):
        c = params["compressor"]
        g = params["gas"]

        self.N = int(c["n_stages"]["value"])
        self.n = c["polytropic_index"]["value"]
        self.eta_p = c["eta_polytropic"]["value"]
        self.eta_m = c["eta_mech"]["value"]
        self.T_inlet_default = c["T_inlet"]["value"]
        self.P_inlet_default = c["P_inlet"]["value"]
        self.P_out_max = c["P_outlet_max"]["value"]

        self.M_NG = g["molar_mass"]["value"]
        self.R_s = g["R_specific"]["value"]
        self.gamma = g["gamma"]["value"]
        self.LHV = g["LHV"]["value"]       # MJ/kg

    def stage_pressure_ratio(self, P_in, P_out):
        P_in = np.asarray(P_in, dtype=float)
        P_out = np.asarray(P_out, dtype=float)
        return (P_out / P_in) ** (1.0 / self.N)

    def stage_discharge_temperature(self, T_in, P_in, P_out):
        """Pre-intercooler discharge temperature per stage [K]."""
        T_in = np.asarray(T_in, dtype=float)
        PRs = self.stage_pressure_ratio(P_in, P_out)
        return T_in * PRs ** ((self.n - 1.0) / self.n)

    def specific_work(self, P_in, P_out, T_in=None):
        """Specific shaft work [J/kg of NG]."""
        T1 = self.T_inlet_default if T_in is None else np.asarray(T_in, dtype=float)
        P_in = np.asarray(P_in, dtype=float)
        P_out = np.asarray(P_out, dtype=float)

        PRs = self.stage_pressure_ratio(P_in, P_out)
        exponent = (self.n - 1.0) / self.n
        w_stage = (self.n / (self.n - 1.0)) * self.R_s * T1 * (PRs ** exponent - 1.0) / self.eta_p
        return self.N * w_stage / self.eta_m

    def specific_work_kJ_per_kg(self, P_in, P_out, T_in=None):
        return self.specific_work(P_in, P_out, T_in) / 1000.0

    def sec_kwh_per_kg(self, P_in, P_out, T_in=None):
        """Specific energy consumption [kWh/kg NG]."""
        return self.specific_work(P_in, P_out, T_in) / 3.6e6

    def shaft_power_kw(self, m_dot, P_in, P_out, T_in=None):
        """Shaft power for given mass flow [kW]."""
        m_dot = np.asarray(m_dot, dtype=float)
        return m_dot * self.specific_work(P_in, P_out, T_in) / 1000.0

    def compression_efficiency(self, P_in, P_out, T_in=None):
        """Ratio LHV / (LHV + specific_work/1e6)."""
        w_MJ = self.specific_work(P_in, P_out, T_in) / 1.0e6
        return self.LHV / (self.LHV + w_MJ)
