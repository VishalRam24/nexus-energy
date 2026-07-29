"""
EC177 — Brushless DC Motor (BLDC) — F1b Efficiency + Thermal

Loss breakdown with temperature dependence:

1. Copper loss (temperature-dependent):
     I = T_mech / k_t(T_m)          [A peak — simplified for surface-PMSM-like BLDC]
     k_t(T_m) = 1.5 * p * Phi_m(T_m)
     Phi_m(T_m) = Phi_m_ref * (1 + alpha_Br * (T_m - T_ref))    [NdFeB]
     R_s(T_w) = R_s_ref * (1 + alpha_Cu * (T_w - T_ref))
     P_copper = I^2 * R_s(T_w)

2. Iron loss (speed-dependent, temperature-independent approximation):
     P_iron = k_iron * omega_rpm^1.5

3. Mechanical loss (friction + windage):
     P_mech = k_mech * omega_rpm

4. Stray loss (proportional to output):
     P_stray = k_stray * P_out

5. Demagnetization risk flag above T_demag.

6. IEC 60034-1 ambient derating above 40C.

References:
    Hanselman, D.C. (2006). Brushless Permanent Magnet Motor Design. Magna Physics.
    Gieras, J.F. (2010). Permanent Magnet Motor Technology, 3rd ed. CRC Press.
    IEC 60034-30-1:2014 (efficiency classes)
"""

import numpy as np

_RPM_TO_RADS = np.pi / 30.0


class BLDCMotorF1b:
    """BLDC motor thermal-dependent loss model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["P_rated_W"]["value"]           # W
        self.T_rated = u["T_rated_Nm"]["value"]          # Nm
        self.omega_rated = u["omega_rated_rpm"]["value"] # rpm
        self.omega_max = u["omega_max_rpm"]["value"]     # rpm
        self.pole_pairs = u["pole_pairs"]["value"]
        self.R_s_ref = u["R_s_ref"]["value"]             # ohm
        self.T_ref = u["T_ref"]["value"]                 # degC
        self.alpha_Cu = u["alpha_Cu"]["value"]           # 1/K
        self.Phi_m_ref = u["Phi_m_ref"]["value"]         # Wb
        self.alpha_Br = u["alpha_Br"]["value"]           # 1/K (negative for NdFeB)
        self.T_demag = u["T_demag"]["value"]             # degC
        self.k_iron = u["k_iron"]["value"]               # W/(rpm^1.5)
        self.k_mech = u["k_mech"]["value"]               # W/rpm
        self.k_stray = u["k_stray"]["value"]             # dimensionless
        self.ambient_threshold = u["ambient_derating_threshold"]["value"]
        self.derating_slope = u["derating_slope"]["value"]

    # --- thermal factors ---

    def pm_flux(self, magnet_temperature):
        """PM flux linkage [Wb] as function of magnet temperature."""
        T_m = np.asarray(magnet_temperature, dtype=float)
        return self.Phi_m_ref * (1.0 + self.alpha_Br * (T_m - self.T_ref))

    def torque_constant(self, magnet_temperature):
        """Torque constant k_t [Nm/A] = 1.5 * p * Phi_m(T)."""
        Phi = self.pm_flux(magnet_temperature)
        return 1.5 * self.pole_pairs * Phi

    def stator_resistance(self, winding_temperature):
        """R_s(T) [ohm] = R_s_ref * (1 + alpha_Cu*(T - T_ref))."""
        T_w = np.asarray(winding_temperature, dtype=float)
        return self.R_s_ref * (1.0 + self.alpha_Cu * (T_w - self.T_ref))

    def demagnetization_risk(self, magnet_temperature):
        """Boolean: True if magnet temperature exceeds T_demag."""
        T_m = np.asarray(magnet_temperature, dtype=float)
        return T_m >= self.T_demag

    def derating_factor(self, ambient_temperature):
        """IEC 60034-1 ambient derating: linear reduction above 40 C."""
        T_a = np.asarray(ambient_temperature, dtype=float)
        derate = 1.0 - self.derating_slope * np.maximum(T_a - self.ambient_threshold, 0.0)
        return np.clip(derate, 0.0, 1.0)

    # --- loss model ---

    def output_power(self, torque_nm, speed_rpm):
        """Shaft output power [W] = T * omega."""
        T = np.asarray(torque_nm, dtype=float)
        omega = np.asarray(speed_rpm, dtype=float) * _RPM_TO_RADS
        return T * omega

    def losses(self, torque_nm, speed_rpm,
               magnet_temperature=80.0, winding_temperature=80.0):
        """
        Loss breakdown [W].

        Returns dict: p_copper_w, p_iron_w, p_mech_w, p_stray_w, p_total_w.
        """
        T = np.asarray(torque_nm, dtype=float)
        omega_rpm = np.asarray(speed_rpm, dtype=float)

        # Stator current estimate
        k_t = self.torque_constant(magnet_temperature)
        k_t_safe = np.where(np.abs(k_t) > 1e-9, k_t, 1e-9)
        I = T / k_t_safe

        # Copper loss
        R_s = self.stator_resistance(winding_temperature)
        P_copper = I ** 2 * R_s

        # Iron loss (eddy + hysteresis, ~speed^1.5)
        P_iron = self.k_iron * np.abs(omega_rpm) ** 1.5

        # Mechanical loss
        P_mech_loss = self.k_mech * np.abs(omega_rpm)

        # Stray loss
        P_out = self.output_power(T, omega_rpm)
        P_stray = self.k_stray * np.abs(P_out)

        P_total = P_copper + P_iron + P_mech_loss + P_stray

        return {
            "p_copper_w": P_copper,
            "p_iron_w": P_iron,
            "p_mech_w": P_mech_loss,
            "p_stray_w": P_stray,
            "p_total_w": P_total,
        }

    def input_power(self, torque_nm, speed_rpm,
                    magnet_temperature=80.0, winding_temperature=80.0):
        """Electrical input power [W]."""
        P_out = self.output_power(torque_nm, speed_rpm)
        loss = self.losses(torque_nm, speed_rpm, magnet_temperature, winding_temperature)
        return P_out + loss["p_total_w"]

    def efficiency(self, torque_nm, speed_rpm,
                   magnet_temperature=80.0, winding_temperature=80.0):
        """Motor efficiency (output/input)."""
        P_out = self.output_power(torque_nm, speed_rpm)
        P_in = self.input_power(torque_nm, speed_rpm, magnet_temperature, winding_temperature)
        eta = np.where(P_in > 1e-6, P_out / P_in, 0.0)
        return np.clip(eta, 0.0, 1.0)

    def phase_current(self, torque_nm, magnet_temperature=80.0):
        """Phase current [A] from torque and k_t(T)."""
        T = np.asarray(torque_nm, dtype=float)
        k_t = self.torque_constant(magnet_temperature)
        k_t_safe = np.where(np.abs(k_t) > 1e-9, k_t, 1e-9)
        return np.abs(T) / k_t_safe
