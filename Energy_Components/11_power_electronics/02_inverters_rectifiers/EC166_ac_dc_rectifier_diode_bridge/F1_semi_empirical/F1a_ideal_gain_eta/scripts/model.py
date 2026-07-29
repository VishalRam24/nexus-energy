"""
EC166 -- Diode Bridge Rectifier (3-Phase Uncontrolled) -- F1a Ideal Gain + Fixed Efficiency

3-Phase uncontrolled (diode) bridge rectifier:
    V_dc_ideal = (3 * sqrt(2) / pi) * V_LL
               = 1.3505 * V_LL_rms

where V_LL is the line-to-line RMS AC voltage.

This is the average DC output of a 3-phase 6-pulse diode bridge (no firing angle).

With fixed efficiency (accounting for diode drops, commutation notches):
    P_out = eta * P_in
    I_dc = P_out / V_dc

Ripple frequency: 6 * f_line (6-pulse rectifier)

References:
    Mohan, N., Undeland, T.M., & Robbins, W.P. (2003).
    Power Electronics: Converters, Applications, and Design. Wiley, 3rd ed.
"""

import numpy as np

# 3*sqrt(2)/pi
_K_3PHASE = 3.0 * np.sqrt(2.0) / np.pi   # = 1.35048...


class DiodeBridgeRectifierF1a:
    """3-Phase Diode Bridge Rectifier -- ideal gain + fixed efficiency."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta = u["eta"]["value"]
        self.P_rated = u["p_rated"]["value"]

    def v_dc_ideal(self, v_ll):
        """V_dc = (3*sqrt(2)/pi) * V_LL_rms  [V]."""
        return _K_3PHASE * np.asarray(v_ll, dtype=float)

    def output_power(self, p_in):
        """P_out = eta * P_in  [W]."""
        return self.eta * np.asarray(p_in, dtype=float)

    def input_power(self, p_out):
        """P_in = P_out / eta  [W]."""
        return np.asarray(p_out, dtype=float) / self.eta

    def dc_current(self, v_ll, p_out):
        """I_dc = P_out / V_dc  [A]."""
        v_dc = self.v_dc_ideal(v_ll)
        safe = np.abs(v_dc) > 1e-9
        return np.where(safe, np.asarray(p_out, dtype=float) / np.where(safe, v_dc, 1.0), 0.0)

    def losses(self, p_in):
        """P_loss = (1 - eta) * P_in  [W]."""
        return (1.0 - self.eta) * np.asarray(p_in, dtype=float)
