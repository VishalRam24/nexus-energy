"""
EC158 — Boost Converter (Step-Up) — F1a Ideal Gain + Loss Model

Ideal voltage conversion:
    V_out = V_in / (1 - D)   =>   D = 1 - V_in / V_out

Input current (power conservation, lossless):
    I_in = I_out / (1 - D) = I_out * V_out / V_in

Conduction losses (boost topology):
    P_cond = I_in^2 * (Rds_on * D + R_L) + I_out * Vd

    Where:
      - I_in^2 * Rds_on * D  : MOSFET conduction loss (on for D fraction)
      - I_in^2 * R_L          : Inductor copper loss (full cycle)
      - I_out * Vd            : Output diode forward loss

Switching losses:
    P_sw = 0.5 * V_out * I_in * (t_on + t_off) * f_sw
    (MOSFET switches at V_out voltage, carries I_in current)

Efficiency:
    P_out = V_out * I_out
    P_in  = P_out + P_cond + P_sw
    eta   = P_out / P_in

Reference:
    Erickson, R.W. & Maksimovic, D. (2020).
    Fundamentals of Power Electronics, 3rd ed. Springer.
"""

import numpy as np


class BoostConverterF1a:
    """Boost converter — ideal gain + conduction/switching losses."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Rds_on = u["Rds_on"]["value"]   # Ohm
        self.R_L = u["R_L"]["value"]         # Ohm
        self.Vd = u["V_diode"]["value"]      # V
        self.t_on = u["t_on"]["value"]       # s
        self.t_off = u["t_off"]["value"]     # s
        self.f_sw = u["f_sw"]["value"]       # Hz

    def duty_cycle(self, v_in, v_out_target):
        """Ideal duty cycle D = 1 - V_in / V_out. Clipped to [0, 0.95]."""
        v_in = np.asarray(v_in, dtype=float)
        v_out = np.asarray(v_out_target, dtype=float)
        D = 1.0 - v_in / np.where(v_out > 0, v_out, 1.0)
        return np.clip(D, 0.0, 0.95)

    def input_current(self, v_in, v_out_target, i_load):
        """
        Input (inductor) current [A] from ideal power conservation:
            I_in = I_out * V_out / V_in  (ideal, lossless)
        """
        v_in = np.asarray(v_in, dtype=float)
        v_out = np.asarray(v_out_target, dtype=float)
        i_out = np.asarray(i_load, dtype=float)
        return i_out * v_out / np.where(v_in > 0, v_in, 1.0)

    def conduction_losses(self, v_in, v_out_target, i_load):
        """
        Conduction losses [W]:
            P_cond = I_in^2 * (Rds_on*D + R_L) + I_out * Vd
        """
        i_in = self.input_current(v_in, v_out_target, i_load)
        D = self.duty_cycle(v_in, v_out_target)
        i_out = np.asarray(i_load, dtype=float)
        return i_in ** 2 * (self.Rds_on * D + self.R_L) + i_out * self.Vd

    def switching_losses(self, v_in, v_out_target, i_load):
        """
        Switching losses [W]:
            P_sw = 0.5 * V_out * I_in * (t_on + t_off) * f_sw
        """
        v_out = np.asarray(v_out_target, dtype=float)
        i_in = self.input_current(v_in, v_out_target, i_load)
        return 0.5 * v_out * i_in * (self.t_on + self.t_off) * self.f_sw

    def output_voltage(self, v_in, v_out_target):
        """Ideal output voltage [V] = V_in / (1 - D)."""
        v_in = np.asarray(v_in, dtype=float)
        D = self.duty_cycle(v_in, v_out_target)
        return v_in / (1.0 - D)

    def efficiency(self, v_in, v_out_target, i_load):
        """Overall efficiency eta = P_out / (P_out + P_loss)."""
        v_out = self.output_voltage(v_in, v_out_target)
        i_out = np.asarray(i_load, dtype=float)
        p_out = v_out * i_out
        p_cond = self.conduction_losses(v_in, v_out_target, i_load)
        p_sw = self.switching_losses(v_in, v_out_target, i_load)
        p_loss = p_cond + p_sw
        p_in = p_out + p_loss
        return np.where(p_in > 0, p_out / p_in, 0.0)
