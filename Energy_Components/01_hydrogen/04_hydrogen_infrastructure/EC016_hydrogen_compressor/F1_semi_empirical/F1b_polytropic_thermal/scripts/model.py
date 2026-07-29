"""
EC016 -- Hydrogen Compressor -- F1b Polytropic Thermal Model

Extends F1a by adding:
1. Variable T_inlet (not assumed fixed at 298 K).
2. Intercooler effectiveness model (real cooler, not perfect).
3. Discharge temperature per stage (pre- and post-intercooler).

Physical model
--------------
Stage pressure ratio (equal split):
    PR_s = (P_out / P_in)^(1/N)

Stage discharge temperature (polytropic compression from T_in_stage):
    T2s = T_in_stage * PR_s^((n-1)/n)           [K]   -- before intercooler

Intercooler model (effectiveness-NTU type, simple single-stream):
    T_out_ic = T2s - eps * (T2s - T_coolant)    [K]
    eps = intercooler_effectiveness (0=no cooling, 1=full cooling to T_coolant)

The inlet temperature for stage k+1 is T_out_ic of stage k.

Specific work for one stage (polytropic, real inlet T):
    w_s = (n/(n-1)) * R_s * T_in_stage * (PR_s^((n-1)/n) - 1) / eta_p   [J/kg]

Total work:
    w_total = (1/eta_mech) * sum_{k=1}^{N} w_s_k

Heat rejected per stage:
    Q_stage = m_dot * cp * (T2s - T_out_ic)                              [W]

References:
    Sdanghi et al. (2019) RSE Rev., 102, 150.
    Bossel (2006) Proc. IEEE, 94(10), 1826.
    Aungier (2000) Centrifugal Compressors, ASME Press: intercooler effectiveness.
"""

import numpy as np


class H2CompressorThermalModel:
    """
    Multistage hydrogen compressor with variable T_inlet and intercooler effectiveness.
    """

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
        self.eps_ic = c["intercooler_effectiveness"]["value"]
        self.T_cool = c["T_coolant"]["value"]

        self.R_s = h["R_specific"]["value"]
        self.gamma = h["gamma"]["value"]
        self.LHV = h["LHV"]["value"]
        self.cp_H2 = h["cp"]["value"]

    # ------------------------------------------------------------------
    # Stage-by-stage analysis
    # ------------------------------------------------------------------

    def stage_pressure_ratio(self, P_in, P_out):
        """Equal stage pressure ratio."""
        return (np.asarray(P_out, dtype=float) / np.asarray(P_in, dtype=float)) ** (1.0 / self.N)

    def stage_temperature_profile(self, P_in, P_out, T_inlet=None, T_coolant=None, eps_ic=None):
        """
        Compute inlet T, discharge T, and post-intercooler T for each stage.

        Parameters
        ----------
        P_in, P_out : float -- inlet/outlet pressures [bar]
        T_inlet     : float -- first-stage inlet temperature [K]
        T_coolant   : float -- intercooler coolant temperature [K]
        eps_ic      : float -- intercooler effectiveness (0-1)

        Returns
        -------
        dict with arrays (length N) for each stage:
            T_in_stage, T_discharge, T_after_ic, w_stage
        """
        T1 = self.T_inlet_default if T_inlet is None else float(T_inlet)
        T_cool = self.T_cool if T_coolant is None else float(T_coolant)
        eps = self.eps_ic if eps_ic is None else float(eps_ic)

        PRs = self.stage_pressure_ratio(P_in, P_out)
        exponent = (self.n - 1.0) / self.n

        T_in_arr   = np.zeros(self.N)
        T_disc_arr = np.zeros(self.N)   # before intercooler
        T_ic_arr   = np.zeros(self.N)   # after intercooler (= inlet of next stage)
        w_arr      = np.zeros(self.N)   # specific work per stage [J/kg]

        T_current = T1
        for k in range(self.N):
            T_in_arr[k]   = T_current
            T2 = T_current * PRs ** exponent           # polytropic discharge T
            T_disc_arr[k] = T2
            w_arr[k]      = (self.n / (self.n - 1.0)) * self.R_s * T_current * (PRs ** exponent - 1.0) / self.eta_p

            if k < self.N - 1:
                # Intercooler: effectiveness model
                T_after = T2 - eps * (T2 - T_cool)
                T_ic_arr[k] = T_after
                T_current = T_after
            else:
                # Last stage: no intercooler
                T_ic_arr[k] = T2

        return {
            "T_in_stage": T_in_arr,
            "T_discharge": T_disc_arr,
            "T_after_ic": T_ic_arr,
            "w_stage_J_kg": w_arr,
        }

    # ------------------------------------------------------------------
    # Aggregate quantities
    # ------------------------------------------------------------------

    def specific_work(self, P_in, P_out, T_inlet=None, T_coolant=None, eps_ic=None):
        """Total specific shaft work [J/kg]."""
        prof = self.stage_temperature_profile(P_in, P_out, T_inlet, T_coolant, eps_ic)
        return prof["w_stage_J_kg"].sum() / self.eta_m

    def sec_kwh_per_kg(self, P_in, P_out, T_inlet=None, T_coolant=None, eps_ic=None):
        """Specific energy consumption [kWh/kg]."""
        return self.specific_work(P_in, P_out, T_inlet, T_coolant, eps_ic) / 3.6e6

    def shaft_power_kw(self, m_dot, P_in, P_out, T_inlet=None, T_coolant=None, eps_ic=None):
        """Shaft power [kW]."""
        return np.asarray(m_dot, dtype=float) * self.specific_work(P_in, P_out, T_inlet, T_coolant, eps_ic) / 1000.0

    def compression_efficiency(self, P_in, P_out, T_inlet=None, T_coolant=None, eps_ic=None):
        """Energy efficiency = LHV / (LHV + w_total/1e6)."""
        w_MJ = self.specific_work(P_in, P_out, T_inlet, T_coolant, eps_ic) / 1.0e6
        return self.LHV / (self.LHV + w_MJ)

    def heat_rejected_kw(self, m_dot, P_in, P_out, T_inlet=None, T_coolant=None, eps_ic=None):
        """
        Total heat rejected in all intercoolers [kW].
        Q = m_dot * cp * (T_discharge - T_after_ic) for each stage.
        """
        prof = self.stage_temperature_profile(P_in, P_out, T_inlet, T_coolant, eps_ic)
        m = float(m_dot)
        # heat rejected from stages 1..N-1 (last stage has no intercooler)
        Q = 0.0
        for k in range(self.N - 1):
            Q += m * self.cp_H2 * (prof["T_discharge"][k] - prof["T_after_ic"][k])
        return Q / 1000.0  # W -> kW

    def final_discharge_temperature(self, P_in, P_out, T_inlet=None, T_coolant=None, eps_ic=None):
        """Temperature of hydrogen after last stage [K]."""
        prof = self.stage_temperature_profile(P_in, P_out, T_inlet, T_coolant, eps_ic)
        return float(prof["T_discharge"][-1])

    def evaluate(self, m_dot, P_in, P_out, T_inlet=None, T_coolant=None, eps_ic=None):
        """
        Full operating-point evaluation.

        Parameters
        ----------
        m_dot   : float -- mass flow rate [kg/s]
        P_in    : float -- inlet pressure [bar]
        P_out   : float -- outlet pressure [bar]
        T_inlet : float or None -- inlet temperature [K]
        T_coolant : float or None -- intercooler coolant temperature [K]
        eps_ic  : float or None -- intercooler effectiveness

        Returns
        -------
        dict
        """
        prof = self.stage_temperature_profile(P_in, P_out, T_inlet, T_coolant, eps_ic)
        w_total = prof["w_stage_J_kg"].sum() / self.eta_m
        P_kw = float(m_dot) * w_total / 1000.0
        SEC = w_total / 3.6e6
        eta_c = self.LHV / (self.LHV + w_total / 1.0e6)
        Q_kw = self.heat_rejected_kw(m_dot, P_in, P_out, T_inlet, T_coolant, eps_ic)
        T_out = float(prof["T_discharge"][-1])

        return {
            "shaft_power_kW":       P_kw,
            "SEC_kWh_kg":           SEC,
            "efficiency":           eta_c,
            "heat_rejected_kW":     Q_kw,
            "T_discharge_final_K":  T_out,
            "stage_profile":        prof,
        }
