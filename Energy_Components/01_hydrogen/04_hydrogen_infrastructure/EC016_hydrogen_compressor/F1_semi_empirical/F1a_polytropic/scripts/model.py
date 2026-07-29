"""
EC016 — Hydrogen Compressor — F1a Multistage Polytropic Model

Specific polytropic work per stage (intercooled to T1):
    w_stage = (n / (n - 1)) * R_specific * T1 * (PR^((n-1)/n) - 1) / eta_p   [J/kg]

For an intercooled multistage compressor with N stages and overall pressure
ratio PR_total = P_out/P_in, the optimum stage pressure ratio is:
    PR_stage = PR_total^(1/N)

Total specific shaft work:
    w_total = N * w_stage / eta_mech                                          [J/kg]

Discharge temperature per stage (before intercooling):
    T2 = T1 * PR_stage^((n-1)/n)                                              [K]

Power:
    P_shaft = m_dot * w_total                                                 [W]

Specific energy consumption:
    SEC = w_total / 3.6e6                                                     [kWh/kg]

Compression ratio relative to LHV:
    eta_comp = LHV / (LHV + w_total/1e6)                                      [-]

References:
    Sdanghi et al. (2019), Renewable & Sustainable Energy Reviews, 102, 150-170.
    Bossel (2006), Proc. IEEE, 94(10), 1826-1837.
"""

import numpy as np


class H2CompressorF1a:
    """Multistage polytropic hydrogen compressor with intercooling."""

    def __init__(self, params: dict):
        c = params["compressor"]
        h = params["hydrogen"]

        self.N = int(c["n_stages"]["value"])
        self.n = c["polytropic_index"]["value"]
        self.eta_p = c["eta_polytropic"]["value"]
        self.eta_m = c["eta_mech"]["value"]
        self.T_inlet_default = c["T_inlet"]["value"]
        self.P_inlet_default = c["P_inlet"]["value"]
        self.P_out_max = c["P_outlet_max"]["value"]

        self.M_H2 = h["molar_mass"]["value"]
        self.R_s = h["R_specific"]["value"]
        self.gamma = h["gamma"]["value"]
        self.LHV = h["LHV"]["value"]                       # MJ/kg

    # ------------------------------------------------------------------ #
    def stage_pressure_ratio(self, P_in, P_out):
        P_in = np.asarray(P_in, dtype=float)
        P_out = np.asarray(P_out, dtype=float)
        return (P_out / P_in) ** (1.0 / self.N)

    def stage_discharge_temperature(self, T_in, P_in, P_out):
        """Pre-intercooler discharge T per stage [K]."""
        T_in = np.asarray(T_in, dtype=float)
        PRs = self.stage_pressure_ratio(P_in, P_out)
        return T_in * PRs ** ((self.n - 1.0) / self.n)

    def specific_work(self, P_in, P_out, T_in=None):
        """
        Specific shaft work [J/kg of H2].

        Polytropic work for N intercooled stages with optimum equal stage ratios.
        """
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
        """Specific energy consumption [kWh/kg]."""
        return self.specific_work(P_in, P_out, T_in) / 3.6e6

    def shaft_power_kw(self, m_dot, P_in, P_out, T_in=None):
        """Shaft power for given mass flow [kW]."""
        m_dot = np.asarray(m_dot, dtype=float)
        return m_dot * self.specific_work(P_in, P_out, T_in) / 1000.0

    def compression_efficiency(self, P_in, P_out, T_in=None):
        """Energy efficiency = LHV / (LHV + w/1e6)."""
        w_MJ = self.specific_work(P_in, P_out, T_in) / 1.0e6
        return self.LHV / (self.LHV + w_MJ)
