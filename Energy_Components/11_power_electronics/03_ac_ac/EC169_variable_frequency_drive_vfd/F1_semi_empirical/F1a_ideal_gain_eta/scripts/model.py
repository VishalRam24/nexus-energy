"""
EC169 -- Variable Frequency Drive (VFD) -- F1a Ideal Gain + Fixed Efficiency

V/Hz control (scalar control) -- simplest VFD operating mode:
    V_out = V_rated * (f_out / f_rated)    for f_out <= f_rated   [constant V/Hz = constant flux]
    V_out = V_rated                         for f_out > f_rated    [field weakening region]

Clipped to [0, V_rated] to avoid over-voltage.

Output frequency range: 0 to 120 Hz (0 to 2x base for many drives).

Power with fixed efficiency:
    P_out = eta * P_in
    Torque at rated load: T = P_out / omega_out  (omega_out = 2*pi*f_out/p_poles)

References:
    Mohan, N., Undeland, T.M., & Robbins, W.P. (2003). Power Electronics. Wiley.
    IEC 61800-9-2: Ecodesign for power drive systems.
"""

import numpy as np


class VFDF1a:
    """VFD (V/Hz control) -- ideal gain + fixed efficiency."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta = u["eta"]["value"]
        self.P_rated = u["p_rated"]["value"]
        self.V_rated = u["v_rated"]["value"]
        self.f_rated = u["f_rated"]["value"]
        self.f_max = u["f_out_max"]["value"]

    def output_voltage(self, f_out):
        """
        V_out (line-to-line RMS) [V]:
            V_out = V_rated * min(f_out/f_rated, 1)
        """
        f_out = np.clip(np.asarray(f_out, dtype=float), 0.0, self.f_max)
        ratio = np.clip(f_out / self.f_rated, 0.0, 1.0)
        return self.V_rated * ratio

    def output_power(self, p_in):
        """P_out = eta * P_in  [W]."""
        return self.eta * np.asarray(p_in, dtype=float)

    def input_power(self, p_out):
        """P_in = P_out / eta  [W]."""
        return np.asarray(p_out, dtype=float) / self.eta

    def losses(self, p_in):
        """P_loss = (1 - eta) * P_in  [W]."""
        return (1.0 - self.eta) * np.asarray(p_in, dtype=float)
