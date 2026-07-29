"""
EC171 -- Cycloconverter -- F1a Ideal Gain + Fixed Efficiency

Cycloconverter ideal output voltage:
    V_out_peak = V_in_peak * cos(alpha)
    V_out_rms  = V_in_rms * cos(alpha)       [fundamental component]

where alpha is the firing angle (0 = max output, pi/2 = zero output).

Output frequency constraint:
    f_out < f_in / 3  (fundamental limit of cycloconverter)

For 3-phase to 3-phase: f_out_max = f_in / 3 ~ 16.7 Hz (for 50 Hz input)

Power with fixed efficiency:
    P_out = eta * P_in

The firing angle also determines the power factor:
    PF_approx = cos(alpha)   (natural commutation)

References:
    Sen, P.C. (1997). Principles of Electric Machines and Power Electronics. Wiley.
    Mohan, N., Undeland, T.M., & Robbins, W.P. (2003). Power Electronics. Wiley.
"""

import numpy as np


class CycloconverterF1a:
    """Cycloconverter -- ideal gain V_out = V_in * cos(alpha) + fixed efficiency."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta = u["eta"]["value"]
        self.P_rated = u["p_rated"]["value"]
        self.V_in = u["v_in_rms"]["value"]
        self.f_in = u["f_in"]["value"]
        self.f_out_max = u["f_out_max"]["value"]
        self.alpha_max = u["alpha_max"]["value"]

    def output_voltage(self, v_in_rms, alpha_rad):
        """V_out_rms = V_in_rms * cos(alpha) -- fundamental component [V]."""
        alpha = np.clip(np.asarray(alpha_rad, dtype=float), 0.0, self.alpha_max)
        return np.asarray(v_in_rms, dtype=float) * np.cos(alpha)

    def power_factor(self, alpha_rad):
        """Approximate input power factor = cos(alpha)."""
        alpha = np.clip(np.asarray(alpha_rad, dtype=float), 0.0, self.alpha_max)
        return np.cos(alpha)

    def output_power(self, p_in):
        """P_out = eta * P_in  [W]."""
        return self.eta * np.asarray(p_in, dtype=float)

    def input_power(self, p_out):
        """P_in = P_out / eta  [W]."""
        return np.asarray(p_out, dtype=float) / self.eta

    def losses(self, p_in):
        """P_loss = (1 - eta) * P_in  [W]."""
        return (1.0 - self.eta) * np.asarray(p_in, dtype=float)
