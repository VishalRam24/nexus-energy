"""
EC158 -- Boost Converter (Step-Up) -- F1b Detailed Semiconductor Loss Model

Extends F1a by decomposing losses into separate physical mechanisms.

Topology: inductor on input side, MOSFET to ground, diode to output.
    D = 1 - V_in / V_out
    I_in = I_out * V_out / V_in = I_out / (1 - D)

MOSFET conduction loss:
    P_cond_mosfet = I_in_rms_on^2 * R_ds_on
    I_in_rms_on   = I_in * sqrt(D)       (MOSFET carries I_in during D fraction)

Diode conduction loss:
    P_cond_diode = I_D_avg * V_f
    I_D_avg      = I_in * (1 - D) = I_out  (diode conducts during 1-D)

Switching loss (MOSFET switches at V_out voltage, carries I_in current):
    P_sw = 0.5 * V_out * I_in * (t_on + t_off) * f_sw

Inductor DCR loss (inductor carries full I_in for entire cycle):
    P_L = I_in^2 * R_L

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


class BoostConverterF1b:
    """Boost converter -- detailed semiconductor loss model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.R_ds_on = u["R_ds_on"]["value"]   # Ohm
        self.V_f = u["V_f"]["value"]            # V
        self.t_on = u["t_on"]["value"]          # s
        self.t_off = u["t_off"]["value"]        # s
        self.f_sw = u["f_sw"]["value"]          # Hz
        self.R_L = u["R_L"]["value"]            # Ohm

    def duty_cycle(self, v_in, v_out_target):
        """Ideal duty cycle D = 1 - V_in / V_out. Clipped to [0, 0.95]."""
        v_in = np.asarray(v_in, dtype=float)
        v_out = np.asarray(v_out_target, dtype=float)
        D = 1.0 - v_in / np.where(v_out > 0, v_out, 1.0)
        return np.clip(D, 0.0, 0.95)

    def output_voltage(self, v_in, v_out_target):
        """Ideal output voltage [V] = V_in / (1 - D)."""
        v_in = np.asarray(v_in, dtype=float)
        D = self.duty_cycle(v_in, v_out_target)
        return v_in / np.where((1.0 - D) > 0.01, 1.0 - D, 0.01)

    def input_current(self, v_in, v_out_target, i_load):
        """Input (inductor) current [A]: I_in = I_out * V_out / V_in."""
        v_in = np.asarray(v_in, dtype=float)
        v_out = np.asarray(v_out_target, dtype=float)
        i_out = np.asarray(i_load, dtype=float)
        return i_out * v_out / np.where(v_in > 0, v_in, 1.0)

    def mosfet_conduction_loss(self, v_in, v_out_target, i_load):
        """MOSFET conduction loss [W]: I_in^2 * D * R_ds_on."""
        D = self.duty_cycle(v_in, v_out_target)
        i_in = self.input_current(v_in, v_out_target, i_load)
        return i_in ** 2 * D * self.R_ds_on

    def diode_conduction_loss(self, v_in, v_out_target, i_load):
        """Diode conduction loss [W]: I_out * V_f (diode carries output current)."""
        i_out = np.asarray(i_load, dtype=float)
        return i_out * self.V_f

    def switching_loss(self, v_in, v_out_target, i_load):
        """Switching loss [W]: 0.5 * V_out * I_in * (t_on + t_off) * f_sw."""
        v_out = np.asarray(v_out_target, dtype=float)
        i_in = self.input_current(v_in, v_out_target, i_load)
        return 0.5 * v_out * i_in * (self.t_on + self.t_off) * self.f_sw

    def inductor_loss(self, v_in, v_out_target, i_load):
        """Inductor DCR loss [W]: I_in^2 * R_L."""
        i_in = self.input_current(v_in, v_out_target, i_load)
        return i_in ** 2 * self.R_L

    def total_losses(self, v_in, v_out_target, i_load):
        """Sum of all loss components [W]."""
        return (self.mosfet_conduction_loss(v_in, v_out_target, i_load) +
                self.diode_conduction_loss(v_in, v_out_target, i_load) +
                self.switching_loss(v_in, v_out_target, i_load) +
                self.inductor_loss(v_in, v_out_target, i_load))

    def loss_breakdown(self, v_in, v_out_target, i_load):
        """Return dict of individual loss components [W]."""
        return {
            "p_mosfet_cond_w": self.mosfet_conduction_loss(v_in, v_out_target, i_load),
            "p_diode_cond_w": self.diode_conduction_loss(v_in, v_out_target, i_load),
            "p_switching_w": self.switching_loss(v_in, v_out_target, i_load),
            "p_inductor_w": self.inductor_loss(v_in, v_out_target, i_load),
        }

    def efficiency(self, v_in, v_out_target, i_load):
        """Overall efficiency eta = P_out / (P_out + P_loss)."""
        v_out = self.output_voltage(v_in, v_out_target)
        i_out = np.asarray(i_load, dtype=float)
        p_out = v_out * i_out
        p_loss = self.total_losses(v_in, v_out_target, i_load)
        p_in = p_out + p_loss
        safe = p_in > 0
        return np.where(safe, p_out / np.where(safe, p_in, 1.0), 0.0)
