"""
EC179 — Wound Rotor Synchronous Generator — F1b Efficiency + Thermal

Loss breakdown for synchronous generator (constant speed, variable load):

1. Stator copper loss (temperature-dependent):
     I_s = P_out / (sqrt(3) * V_line * pf)    [stator current]
     R_s(T_s) = R_s_ref * (1 + alpha_Cu * (T_s - T_ref))
     P_stator_cu = 3 * I_s^2 * R_s(T_s)

2. Rotor (field) excitation copper loss (temperature-dependent):
     P_rotor_cu = I_f^2 * R_f(T_r)
     R_f(T_r) = R_f_ref * (1 + alpha_Cu * (T_r - T_ref))
     I_f ~ I_f_rated * sqrt(1 + (load_fraction * tan(acos(pf)))^2)
     (simplified: excitation scales with reactive demand)

3. Iron loss (constant at synchronous speed, ~constant for synchronous machine):
     P_iron = k_iron   [fixed]

4. Mechanical loss (friction + windage at synchronous speed):
     P_mech = k_mech   [fixed]

5. Stray loss:
     P_stray = k_stray * P_rated

Generator efficiency:
     eta = P_out / (P_out + P_stator_cu + P_rotor_cu + P_iron + P_mech + P_stray)

References:
    Kundur, P. (1994). Power System Stability and Control. McGraw-Hill.
    Chapman, S.J. (2012). Electric Machinery Fundamentals, 5th ed. McGraw-Hill.
    IEC 60034-30-1:2014.
"""

import numpy as np


class WRSGF1b:
    """Wound Rotor Synchronous Generator — thermal-dependent efficiency model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.S_rated = u["S_rated_kVA"]["value"] * 1000.0  # VA
        self.P_rated = u["P_rated_kW"]["value"] * 1000.0   # W
        self.V_line = u["V_line_kV"]["value"] * 1000.0     # V
        self.pf_rated = u["pf_rated"]["value"]
        self.omega_sync = u["omega_sync_rpm"]["value"]
        self.R_s_ref = u["R_s_ref"]["value"]
        self.R_f_ref = u["R_f_ref"]["value"]
        self.T_ref = u["T_ref"]["value"]
        self.alpha_Cu = u["alpha_Cu"]["value"]
        self.I_f_rated = u["I_f_rated"]["value"]
        self.k_iron = u["k_iron"]["value"]      # W (constant)
        self.k_mech = u["k_mech"]["value"]      # W (constant)
        self.k_stray = u["k_stray"]["value"]
        self.ambient_threshold = u["ambient_derating_threshold"]["value"]
        self.derating_slope = u["derating_slope"]["value"]

    def stator_resistance(self, stator_temperature):
        T_s = np.asarray(stator_temperature, dtype=float)
        return self.R_s_ref * (1.0 + self.alpha_Cu * (T_s - self.T_ref))

    def field_resistance(self, rotor_temperature):
        T_r = np.asarray(rotor_temperature, dtype=float)
        return self.R_f_ref * (1.0 + self.alpha_Cu * (T_r - self.T_ref))

    def stator_current(self, load_fraction, power_factor=None):
        """
        Stator current [A] from load and terminal voltage.
            I_s = PLR * P_rated / (sqrt(3) * V_line * pf)
        """
        plr = np.asarray(load_fraction, dtype=float)
        pf = self.pf_rated if power_factor is None else np.asarray(power_factor, dtype=float)
        denom = np.sqrt(3.0) * self.V_line * pf
        return plr * self.P_rated / denom

    def field_current(self, load_fraction, power_factor=None):
        """
        Approximate field current [A].
        At rated load and pf: I_f = I_f_rated.
        Reactive component increases I_f above unity power factor:
            I_f ~ I_f_rated * PLR * sqrt(1 + tan^2(phi))
            where phi = acos(pf)
        """
        plr = np.asarray(load_fraction, dtype=float)
        pf = self.pf_rated if power_factor is None else np.asarray(power_factor, dtype=float)
        pf_clip = np.clip(pf, 0.01, 1.0)
        tan_phi = np.sqrt(np.maximum(1.0 - pf_clip**2, 0.0)) / pf_clip
        return self.I_f_rated * plr * np.sqrt(1.0 + tan_phi**2)

    def losses(self, load_fraction, stator_temperature=75.0,
               rotor_temperature=75.0, power_factor=None):
        """
        Full loss breakdown [W].

        Returns: p_stator_cu_w, p_rotor_cu_w, p_iron_w, p_mech_w, p_stray_w, p_total_w.
        """
        plr = np.asarray(load_fraction, dtype=float)

        R_s = self.stator_resistance(stator_temperature)
        R_f = self.field_resistance(rotor_temperature)
        I_s = self.stator_current(plr, power_factor)
        I_f = self.field_current(plr, power_factor)

        P_stator_cu = 3.0 * I_s**2 * R_s
        P_rotor_cu = I_f**2 * R_f
        P_iron = np.full_like(plr, self.k_iron, dtype=float)
        P_mech = np.full_like(plr, self.k_mech, dtype=float)
        P_stray = self.k_stray * self.P_rated * np.ones_like(plr)

        P_total = P_stator_cu + P_rotor_cu + P_iron + P_mech + P_stray

        return {
            "p_stator_cu_w": P_stator_cu,
            "p_rotor_cu_w": P_rotor_cu,
            "p_iron_w": P_iron,
            "p_mech_w": P_mech,
            "p_stray_w": P_stray,
            "p_total_w": P_total,
        }

    def output_power(self, load_fraction):
        """Active power output [W]."""
        plr = np.asarray(load_fraction, dtype=float)
        return plr * self.P_rated

    def input_power(self, load_fraction, stator_temperature=75.0,
                    rotor_temperature=75.0, power_factor=None):
        """Mechanical shaft input power [W] = P_out + losses."""
        P_out = self.output_power(load_fraction)
        loss = self.losses(load_fraction, stator_temperature, rotor_temperature, power_factor)
        return P_out + loss["p_total_w"]

    def efficiency(self, load_fraction, stator_temperature=75.0,
                   rotor_temperature=75.0, power_factor=None):
        """Generator efficiency = P_out / P_in."""
        P_out = self.output_power(load_fraction)
        P_in = self.input_power(load_fraction, stator_temperature, rotor_temperature, power_factor)
        eta = np.where(P_in > 1.0, P_out / P_in, 0.0)
        return np.clip(eta, 0.0, 1.0)

    def derating_factor(self, ambient_temperature):
        """IEC 60034-1 ambient derating above 40 C."""
        T_a = np.asarray(ambient_temperature, dtype=float)
        d = 1.0 - self.derating_slope * np.maximum(T_a - self.ambient_threshold, 0.0)
        return np.clip(d, 0.0, 1.0)
