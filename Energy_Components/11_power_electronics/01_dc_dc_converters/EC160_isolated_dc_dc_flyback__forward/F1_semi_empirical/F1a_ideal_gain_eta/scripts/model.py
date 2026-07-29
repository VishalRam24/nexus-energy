"""
EC160 -- Isolated DC-DC Converter (Flyback/Forward) -- F1a Ideal Gain + Fixed Efficiency

Flyback topology ideal voltage conversion:
    V_out = N * D * V_in

where N = N2/N1 (secondary-to-primary turns ratio), D = duty cycle.

Power transfer (fixed efficiency):
    P_out = eta * P_in
    P_in  = P_out / eta

Reference:
    Erickson, R.W. & Maksimovic, D. (2020).
    Fundamentals of Power Electronics, 3rd ed. Springer.
"""

import numpy as np


class IsolatedDCDCF1a:
    """Isolated DC-DC (Flyback/Forward) -- ideal gain + fixed efficiency."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.N = u["n_turns"]["value"]      # turns ratio N2/N1
        self.eta = u["eta"]["value"]        # fixed efficiency
        self.P_rated = u["p_rated"]["value"]  # W
        self.d_max = u["d_max"]["value"]

    def output_voltage(self, v_in, duty_cycle):
        """V_out = N * D * V_in  [V]."""
        v_in = np.asarray(v_in, dtype=float)
        D = np.clip(np.asarray(duty_cycle, dtype=float), 0.0, self.d_max)
        return self.N * D * v_in

    def duty_cycle_for_vout(self, v_in, v_out_target):
        """Compute D = V_out / (N * V_in). Clipped to [0.05, d_max]."""
        v_in = np.asarray(v_in, dtype=float)
        v_out = np.asarray(v_out_target, dtype=float)
        safe = np.abs(v_in) > 1e-12
        denom = self.N * np.where(safe, v_in, 1.0)
        D = np.where(safe, v_out / denom, 0.0)
        return np.clip(D, 0.05, self.d_max)

    def output_power(self, p_in):
        """P_out = eta * P_in  [W]."""
        return self.eta * np.asarray(p_in, dtype=float)

    def input_power(self, p_out):
        """P_in = P_out / eta  [W]."""
        return np.asarray(p_out, dtype=float) / self.eta

    def losses(self, p_in):
        """P_loss = (1 - eta) * P_in  [W]."""
        return (1.0 - self.eta) * np.asarray(p_in, dtype=float)
