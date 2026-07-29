"""
EC167 -- Active Front End Rectifier with PFC -- F1a Ideal Gain + Fixed Efficiency

Active Front End (AFE) / Boost PFC Rectifier:
- DC output voltage is actively controlled (set point), independent of AC input
  (within the controllable range: V_dc > sqrt(2) * V_LL for boost operation)
- Unity power factor: I_ac draws sinusoidal current in phase with V_ac
- Low THD (< 5% typical for high-switching AFE)

Input power from AC (unity PF, per phase):
    P_in = 3 * V_phase_rms * I_rms * cos(phi)  with cos(phi) = PF = 1
         = sqrt(3) * V_LL * I_rms * PF

Output DC:
    V_dc = V_dc_setpoint   (controlled by feedback)
    P_out = eta * P_in
    I_dc = P_out / V_dc

AC apparent input current (per phase):
    S = P_in  (since PF=1)
    I_ac_rms = P_in / (sqrt(3) * V_LL)

References:
    Mohan, N., Undeland, T.M., & Robbins, W.P. (2003).
    Power Electronics: Converters, Applications, and Design. Wiley.
"""

import numpy as np


class AFEPFCRectifierF1a:
    """Active Front End Rectifier with PFC -- controlled V_dc + fixed efficiency."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta = u["eta"]["value"]
        self.P_rated = u["p_rated"]["value"]
        self.pf = u["pf_nominal"]["value"]   # ~1.0
        self.thd_i = u["thd_i"]["value"]
        self.v_dc_set_default = u["v_dc_set"]["value"]

    def output_voltage(self, v_dc_set=None):
        """V_dc = setpoint (controlled). Returns v_dc_set."""
        if v_dc_set is None:
            return self.v_dc_set_default
        return np.asarray(v_dc_set, dtype=float)

    def output_power(self, p_in):
        """P_out = eta * P_in  [W]."""
        return self.eta * np.asarray(p_in, dtype=float)

    def input_power(self, p_out):
        """P_in = P_out / eta  [W]."""
        return np.asarray(p_out, dtype=float) / self.eta

    def ac_current_rms(self, v_ll, p_in):
        """I_ac_rms (per phase) = P_in / (sqrt(3) * V_LL)  [A]."""
        v_ll = np.asarray(v_ll, dtype=float)
        p_in = np.asarray(p_in, dtype=float)
        safe = v_ll > 1e-6
        return np.where(safe, p_in / (np.sqrt(3.0) * np.where(safe, v_ll, 1.0)), 0.0)

    def dc_current(self, v_dc_set, p_out):
        """I_dc = P_out / V_dc  [A]."""
        v_dc = np.asarray(v_dc_set, dtype=float)
        p_out = np.asarray(p_out, dtype=float)
        safe = v_dc > 1e-6
        return np.where(safe, p_out / np.where(safe, v_dc, 1.0), 0.0)

    def losses(self, p_in):
        """P_loss = (1 - eta) * P_in  [W]."""
        return (1.0 - self.eta) * np.asarray(p_in, dtype=float)
