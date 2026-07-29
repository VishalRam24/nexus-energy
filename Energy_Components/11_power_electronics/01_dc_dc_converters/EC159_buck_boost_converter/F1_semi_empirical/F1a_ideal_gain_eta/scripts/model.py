"""
EC159 -- Buck-Boost Converter -- F1a Ideal Gain + Efficiency Model

Ideal conversion law for the inverting buck-boost topology:
    V_out = D / (1 - D) * V_in    [magnitude; output polarity is inverted]

With constant efficiency eta:
    P_out = eta * P_in
    I_out = P_out / V_out  (from load)
    I_in  = P_in / V_in

Duty cycle D bounded to [0.1, 0.9] for practical operation.

F1b adds detailed semiconductor and inductor loss models.

References:
    Erickson, R.W. & Maksimovic, D. (2001). Fundamentals of Power Electronics, 2nd ed.
    Kazimierczuk, M.K. (2015). Pulse-Width Modulated DC-DC Power Converters, 2nd ed.
"""

import numpy as np


class BuckBoostConverterF1a:
    """Buck-boost converter: ideal gain with constant efficiency."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta   = float(u["eta"]["value"])
        self.D_min = float(u["D_min"]["value"])
        self.D_max = float(u["D_max"]["value"])

    def duty_cycle_for_voltage(self, v_in: float, v_out_mag: float) -> float:
        """
        Compute duty cycle D to achieve |V_out| from V_in.
            V_out_mag = D/(1-D) * V_in  =>  D = V_out_mag / (V_out_mag + V_in)
        """
        v_in     = float(np.clip(v_in, 1e-6, None))
        v_out_m  = float(np.clip(v_out_mag, 1e-6, None))
        D = v_out_m / (v_out_m + v_in)
        return float(np.clip(D, self.D_min, self.D_max))

    def predict(self, v_in: float, D: float, P_out: float = 0.0) -> dict:
        """
        Parameters
        ----------
        v_in  : float  Input voltage [V]
        D     : float  Duty cycle [0.1 – 0.9]
        P_out : float  Output power demand [W] (optional, for current calc)

        Returns
        -------
        dict with:
            D_clamped     : effective duty cycle after clamping
            voltage_gain  : |V_out| / V_in = D/(1-D)
            V_out_mag     : output voltage magnitude [V]
            V_in          : input voltage [V]
            eta           : efficiency [-]
            P_in          : input power [W] (if P_out provided)
            I_out         : output current [A] (if P_out provided)
            I_in          : input current [A] (if P_out provided)
        """
        v_in = float(np.clip(v_in, 1e-6, None))
        D    = float(np.clip(D, self.D_min, self.D_max))

        gain   = D / (1.0 - D)
        V_out  = gain * v_in

        # Power calculations (if load specified)
        if P_out > 0:
            P_in  = float(P_out) / self.eta
            I_out = float(P_out) / V_out
            I_in  = P_in / v_in
        else:
            P_in = I_out = I_in = 0.0

        return {
            "D_clamped":    float(D),
            "voltage_gain": float(gain),
            "V_out_mag":    float(V_out),
            "V_in":         float(v_in),
            "eta":          float(self.eta),
            "P_in_W":       float(P_in),
            "P_out_W":      float(P_out),
            "I_out_A":      float(I_out),
            "I_in_A":       float(I_in),
        }
