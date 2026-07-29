"""
EC157 -- Buck Converter (Step-Down) -- F1b Detailed Semiconductor Loss Model

Extends F1a by decomposing losses into separate physical mechanisms:

MOSFET conduction loss:
    P_cond_mosfet = I_rms_mosfet^2 * R_ds_on
    I_rms_mosfet  = I_out * sqrt(D)        (current flows during D fraction)

Diode conduction loss:
    P_cond_diode = I_D_avg * V_f
    I_D_avg      = I_out * (1 - D)         (diode conducts during 1-D)

Switching loss (MOSFET hard-switching transitions):
    P_sw = 0.5 * V_in * I_out * (t_on + t_off) * f_sw

Inductor DCR loss:
    P_L = I_rms_L^2 * R_L
    I_rms_L ~ I_out                        (assuming small ripple)

Total losses:
    P_loss = P_cond_mosfet + P_cond_diode + P_sw + P_L

Efficiency:
    P_out = V_out * I_out
    P_in  = P_out + P_loss
    eta   = P_out / P_in

Reference:
    Erickson, R.W. & Maksimovic, D. (2020).
    Fundamentals of Power Electronics, 3rd ed. Springer.
"""

import numpy as np


class BuckConverterF1b:
    """Buck converter -- detailed semiconductor loss model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.R_ds_on = u["R_ds_on"]["value"]   # Ohm
        self.V_f = u["V_f"]["value"]            # V
        self.t_on = u["t_on"]["value"]          # s
        self.t_off = u["t_off"]["value"]        # s
        self.f_sw = u["f_sw"]["value"]          # Hz
        self.R_L = u["R_L"]["value"]            # Ohm

    def duty_cycle(self, v_in, v_out_target):
        """Ideal duty cycle D = V_out / V_in."""
        v_in = np.asarray(v_in, dtype=float)
        v_out = np.asarray(v_out_target, dtype=float)
        safe = np.abs(v_in) > 1e-12
        D = np.where(safe, np.clip(v_out / np.where(safe, v_in, 1.0), 0.0, 1.0), 0.0)
        return D

    def output_voltage(self, v_in, v_out_target):
        """Ideal output voltage [V] = D * V_in."""
        return self.duty_cycle(v_in, v_out_target) * np.asarray(v_in, dtype=float)

    def mosfet_conduction_loss(self, v_in, v_out_target, i_load):
        """MOSFET conduction loss [W]: I_rms^2 * R_ds_on, where I_rms = I_out * sqrt(D)."""
        D = self.duty_cycle(v_in, v_out_target)
        i = np.asarray(i_load, dtype=float)
        i_rms_sq = i ** 2 * D  # I_rms^2 = I_out^2 * D
        return i_rms_sq * self.R_ds_on

    def diode_conduction_loss(self, v_in, v_out_target, i_load):
        """Diode conduction loss [W]: I_D_avg * V_f, where I_D_avg = I_out * (1-D)."""
        D = self.duty_cycle(v_in, v_out_target)
        i = np.asarray(i_load, dtype=float)
        i_d_avg = i * (1.0 - D)
        return i_d_avg * self.V_f

    def switching_loss(self, v_in, i_load):
        """Switching loss [W]: 0.5 * V_in * I_out * (t_on + t_off) * f_sw."""
        v_in = np.asarray(v_in, dtype=float)
        i = np.asarray(i_load, dtype=float)
        return 0.5 * v_in * i * (self.t_on + self.t_off) * self.f_sw

    def inductor_loss(self, i_load):
        """Inductor DCR loss [W]: I_out^2 * R_L (inductor carries full current)."""
        i = np.asarray(i_load, dtype=float)
        return i ** 2 * self.R_L

    def total_losses(self, v_in, v_out_target, i_load):
        """Sum of all loss components [W]."""
        p_mosfet = self.mosfet_conduction_loss(v_in, v_out_target, i_load)
        p_diode = self.diode_conduction_loss(v_in, v_out_target, i_load)
        p_sw = self.switching_loss(v_in, i_load)
        p_ind = self.inductor_loss(i_load)
        return p_mosfet + p_diode + p_sw + p_ind

    def loss_breakdown(self, v_in, v_out_target, i_load):
        """Return dict of individual loss components [W]."""
        return {
            "p_mosfet_cond_w": self.mosfet_conduction_loss(v_in, v_out_target, i_load),
            "p_diode_cond_w": self.diode_conduction_loss(v_in, v_out_target, i_load),
            "p_switching_w": self.switching_loss(v_in, i_load),
            "p_inductor_w": self.inductor_loss(i_load),
        }

    def efficiency(self, v_in, v_out_target, i_load):
        """Overall efficiency eta = P_out / (P_out + P_loss)."""
        v_out = self.output_voltage(v_in, v_out_target)
        i = np.asarray(i_load, dtype=float)
        p_out = v_out * i
        p_loss = self.total_losses(v_in, v_out_target, i_load)
        p_in = p_out + p_loss
        safe = p_in > 0
        return np.where(safe, p_out / np.where(safe, p_in, 1.0), 0.0)
