"""
EC165 -- Multilevel Inverter (3-Level NPC) -- F1a Ideal Gain + Fixed Efficiency

3-Level NPC output voltage (peak phase):
    V_ac_peak = ma * V_dc / 2

RMS phase voltage:
    V_ac_rms = ma * V_dc / (2 * sqrt(2))

Line-to-line RMS:
    V_LL_rms = sqrt(3) * V_ac_rms = ma * sqrt(3) * V_dc / (2 * sqrt(2))

Modulation index ma in [0, 1.15] (overmodulation extends to 1.15 -> square wave).
THD approximate (fundamental dominated):
    THD ~ k_thd / ma  (NPC multilevel reduces lower-order harmonics vs 2-level)
    Using simplified: THD_approx = 0.02 / ma (very low for NPC 3-level)

Fixed efficiency: P_out = eta * P_in

References:
    Rodriguez, J., Lai, J.S., & Peng, F.Z. (2002).
    IEEE Trans. Ind. Electron., 49(4), 724-738.
"""

import numpy as np


class MultilevelInverterF1a:
    """3-Level NPC Multilevel Inverter -- ideal gain + fixed efficiency."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta = u["eta"]["value"]
        self.P_rated = u["p_rated"]["value"]
        self.n_levels = u["n_levels"]["value"]
        # THD coefficient for 3-level NPC (simplified first-harmonic model)
        self._k_thd = 0.02

    def v_ac_rms_phase(self, v_dc, ma):
        """RMS phase voltage [V] = ma * V_dc / (2*sqrt(2))."""
        v_dc = np.asarray(v_dc, dtype=float)
        ma = np.clip(np.asarray(ma, dtype=float), 0.0, 1.15)
        return ma * v_dc / (2.0 * np.sqrt(2.0))

    def v_ac_rms_line(self, v_dc, ma):
        """RMS line-to-line voltage [V] = sqrt(3) * V_phase."""
        return np.sqrt(3.0) * self.v_ac_rms_phase(v_dc, ma)

    def thd_approx(self, ma):
        """Approximate THD (fractional). Lower at higher ma for NPC."""
        ma = np.clip(np.asarray(ma, dtype=float), 0.01, 1.15)
        return self._k_thd / ma

    def output_power(self, p_in):
        """P_out = eta * P_in  [W]."""
        return self.eta * np.asarray(p_in, dtype=float)

    def losses(self, p_in):
        """P_loss = (1 - eta) * P_in  [W]."""
        return (1.0 - self.eta) * np.asarray(p_in, dtype=float)
