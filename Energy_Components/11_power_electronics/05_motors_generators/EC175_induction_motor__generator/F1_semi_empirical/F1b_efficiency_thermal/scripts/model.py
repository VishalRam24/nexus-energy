"""
EC175 — Induction Motor/Generator — F1b Efficiency + Thermal

Extends F1a (two-component loss model) with temperature-dependent copper resistance:
    R(T) = R_ref * (1 + alpha_Cu * (T_winding - T_ref))

The variable (copper) losses scale with resistance ratio:
    c2(T) = c2_ref * R(T) / R_ref = c2_ref * (1 + alpha_Cu * (T - T_ref))

Constant losses (iron + friction/windage) are assumed temperature-independent.

Derating per IEC 60034-1: above 40C ambient, output must be derated to prevent
winding insulation damage. Linear derating ~1%/K above threshold.

Current estimation:
    I = P_in / (sqrt(3) * V_line * pf)

References:
    IEC 60034-30-1:2014 (efficiency classes)
    IEC 60034-1:2022 (thermal derating)
    Boldea, I. & Nasar, S.A. (2010). The Induction Machine Handbook. CRC Press.
"""

import numpy as np


class InductionMotorF1b:
    """Induction motor/generator — thermal-dependent efficiency model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["rated_power_kw"]["value"]         # kW
        self.eta_rated = u["eta_rated"]["value"]
        self.pf = u["power_factor"]["value"]
        self.n_sync = u["sync_speed_rpm"]["value"]           # rpm
        self.n_rated = u["rated_speed_rpm"]["value"]         # rpm
        self.s_rated = u["rated_slip"]["value"]
        self.V_line = u["V_line"]["value"]                   # V
        self.R_ref = u["R_ref"]["value"]                     # ohm at T_ref
        self.T_ref = u["T_ref"]["value"]                     # degC
        self.alpha_Cu = u["alpha_Cu"]["value"]               # 1/K
        self.ambient_threshold = u["ambient_derating_threshold"]["value"]  # degC
        self.derating_slope = u["derating_slope"]["value"]   # 1/K

        # Two-component loss model at reference temperature
        total_loss_frac = 1.0 / self.eta_rated - 1.0
        self.c0 = u["constant_loss_fraction"]["value"] * total_loss_frac
        self.c2_ref = u["variable_loss_fraction"]["value"] * total_loss_frac

    def resistance_ratio(self, winding_temperature):
        """R(T) / R_ref = 1 + alpha_Cu * (T - T_ref)."""
        T = np.asarray(winding_temperature, dtype=float)
        return 1.0 + self.alpha_Cu * (T - self.T_ref)

    def derating_factor(self, ambient_temperature):
        """
        IEC 60034-1 derating: full rating up to 40C ambient,
        linear derating above that.
        """
        T_amb = np.asarray(ambient_temperature, dtype=float)
        derate = 1.0 - self.derating_slope * np.maximum(T_amb - self.ambient_threshold, 0.0)
        return np.clip(derate, 0.0, 1.0)

    def efficiency(self, plr, winding_temperature=75.0):
        """
        Efficiency with temperature-dependent copper losses.
            c2(T) = c2_ref * R(T)/R_ref
            eta = PLR / (PLR + c0 + c2(T)*PLR^2)
        """
        plr = np.asarray(plr, dtype=float)
        R_ratio = self.resistance_ratio(winding_temperature)
        c2_T = self.c2_ref * R_ratio
        eta = plr / (plr + self.c0 + c2_T * plr ** 2)
        return np.clip(eta, 1e-6, 0.9999)

    def output_power(self, plr, ambient_temperature=25.0):
        """Mechanical output power [kW], derated if ambient > 40C."""
        plr = np.asarray(plr, dtype=float)
        derate = self.derating_factor(ambient_temperature)
        return plr * self.P_rated * derate

    def input_power(self, plr, winding_temperature=75.0, ambient_temperature=25.0):
        """Electrical input power [kW]."""
        eta = self.efficiency(plr, winding_temperature)
        p_out = self.output_power(plr, ambient_temperature)
        return np.where(eta > 0, p_out / eta, 0.0)

    def losses(self, plr, winding_temperature=75.0, ambient_temperature=25.0):
        """Total losses [kW] = P_in - P_out."""
        return (self.input_power(plr, winding_temperature, ambient_temperature)
                - self.output_power(plr, ambient_temperature))

    def current(self, plr, winding_temperature=75.0, ambient_temperature=25.0):
        """Line current [A] = P_in / (sqrt(3) * V_line * pf)."""
        p_in = self.input_power(plr, winding_temperature, ambient_temperature)
        return p_in * 1000.0 / (np.sqrt(3) * self.V_line * self.pf)

    def slip(self, plr=1.0):
        """Approximate slip: s ~ s_rated * PLR."""
        plr = np.asarray(plr, dtype=float)
        return np.clip(self.s_rated * plr, 0.0, 1.0)

    def rotor_speed(self, plr=1.0):
        """Rotor speed [rpm]."""
        s = self.slip(plr)
        return self.n_sync * (1.0 - s)
