"""
EC157 — Buck Converter (Step-Down) — F1a Ideal Gain + Loss Model

Ideal voltage conversion:
    V_out = D * V_in           (D = duty cycle)
    D = V_out_target / V_in

Conduction losses:
    P_cond = I_out^2 * (Rds_on * D + R_L + Vd * (1-D) / V_out)
    Note: Vd*(1-D)/V_out term is the effective diode resistance contribution

Switching losses (MOSFET transitions):
    P_sw = 0.5 * V_in * I_out * (t_on + t_off) * f_sw

Total losses:
    P_loss = P_cond + P_sw

Efficiency:
    P_out = V_out * I_out
    P_in  = P_out + P_loss
    eta   = P_out / P_in

Reference:
    Erickson, R.W. & Maksimovic, D. (2020).
    Fundamentals of Power Electronics, 3rd ed. Springer.
"""

import numpy as np


class BuckConverterF1a:
    """Buck converter — ideal gain + conduction/switching losses."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Rds_on = u["Rds_on"]["value"]      # Ohm
        self.R_L = u["R_L"]["value"]            # Ohm
        self.Vd = u["V_diode"]["value"]         # V
        self.t_on = u["t_on"]["value"]          # s
        self.t_off = u["t_off"]["value"]        # s
        self.f_sw = u["f_sw"]["value"]          # Hz

    def duty_cycle(self, v_in, v_out_target):
        """Ideal duty cycle D = V_out / V_in.  Returns 0 when V_in ~ 0."""
        v_in = np.asarray(v_in, dtype=float)
        v_out = np.asarray(v_out_target, dtype=float)
        safe = np.abs(v_in) > 1e-12
        D = np.where(safe, np.clip(v_out / np.where(safe, v_in, 1.0), 0.0, 1.0), 0.0)
        return D

    def conduction_losses(self, v_in, v_out_target, i_load):
        """
        Conduction losses [W]:
            P_cond = I_out^2 * (Rds_on*D + R_L + Vd*(1-D)/V_out)
        """
        v_in = np.asarray(v_in, dtype=float)
        v_out = np.asarray(v_out_target, dtype=float)
        i = np.asarray(i_load, dtype=float)
        D = self.duty_cycle(v_in, v_out)
        R_eff = self.Rds_on * D + self.R_L + self.Vd * (1.0 - D) / np.where(v_out > 0, v_out, 1.0)
        # No conduction losses when input voltage is absent
        return np.where(np.abs(v_in) > 1e-12, i ** 2 * R_eff, 0.0)

    def switching_losses(self, v_in, i_load):
        """
        Switching losses [W]:
            P_sw = 0.5 * V_in * I_out * (t_on + t_off) * f_sw
        """
        v_in = np.asarray(v_in, dtype=float)
        i = np.asarray(i_load, dtype=float)
        return 0.5 * v_in * i * (self.t_on + self.t_off) * self.f_sw

    def output_voltage(self, v_in, v_out_target):
        """Ideal output voltage [V] = D * V_in."""
        return self.duty_cycle(v_in, v_out_target) * np.asarray(v_in, dtype=float)

    def efficiency(self, v_in, v_out_target, i_load):
        """Overall efficiency eta = P_out / (P_out + P_loss)."""
        v_out = self.output_voltage(v_in, v_out_target)
        i = np.asarray(i_load, dtype=float)
        p_out = v_out * i
        p_cond = self.conduction_losses(v_in, v_out_target, i_load)
        p_sw = self.switching_losses(v_in, i_load)
        p_loss = p_cond + p_sw
        p_in = p_out + p_loss
        safe = p_in > 0
        return np.where(safe, p_out / np.where(safe, p_in, 1.0), 0.0)
