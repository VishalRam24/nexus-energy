"""
EC178 — Switched Reluctance Motor (SRM) — F1b Efficiency + Thermal

SRM has no permanent magnets — all torque from reluctance variation.
Loss components:
1. Copper loss (temperature-dependent):
     I_ph = T / (k_t_eff * n_phases * delta)    — simplified phase current
     k_t_eff ~ rated efficiency proxy: use energy balance at rated point
     R_ph(T) = R_ph_ref * (1 + alpha_Cu * (T - T_ref))
     P_copper = n_phases * I_ph^2 * R_ph(T)

   Simpler analytically: use total input/output energy balance.
   I_ph estimated from output power:
     I_ph = sqrt(P_out / (n_phases * R_ph_ref))   — at rated efficiency

   More physical approach (used here):
     - Two-component model: P_copper_fraction scales as PLR^2, P_iron as PLR^0
     - c2(T) = c2_ref * (1 + alpha_Cu * (T - T_ref))   — copper scales with R

2. Iron loss (eddy + hysteresis, speed-dependent):
     P_iron = k_iron_eddy * omega_rpm^2 + k_iron_hyst * omega_rpm
     (SRM has significant iron loss due to pulsed flux in stator poles)

3. Mechanical loss:
     P_mech = k_mech * omega_rpm

4. Stray loss:
     P_stray = k_stray * P_out

Efficiency:
     eta = P_out / (P_out + P_copper + P_iron + P_mech + P_stray)

References:
    Miller, T.J.E. (1993). Switched Reluctance Motors and their Control. Magna Physics.
    Krishnan, R. (2001). Switched Reluctance Motor Drives. CRC Press.
    IEC 60034-30-1:2014.
"""

import numpy as np

_RPM_TO_RADS = np.pi / 30.0


class SRMMotorF1b:
    """SRM efficiency model with loss separation and temperature-dependent copper."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["P_rated_W"]["value"]
        self.T_rated = u["T_rated_Nm"]["value"]
        self.omega_rated = u["omega_rated_rpm"]["value"]
        self.omega_max = u["omega_max_rpm"]["value"]
        self.R_ph_ref = u["R_ph_ref"]["value"]
        self.T_ref = u["T_ref"]["value"]
        self.alpha_Cu = u["alpha_Cu"]["value"]
        self.n_phases = int(u["n_phases"]["value"])
        self.k_iron_eddy = u["k_iron_eddy"]["value"]
        self.k_iron_hyst = u["k_iron_hyst"]["value"]
        self.k_mech = u["k_mech"]["value"]
        self.k_stray = u["k_stray"]["value"]
        self.ambient_threshold = u["ambient_derating_threshold"]["value"]
        self.derating_slope = u["derating_slope"]["value"]

        # Calibrate copper coefficient at rated conditions (reference T=T_ref)
        # Estimate rated I_ph from rated power and resistance
        # P_copper_rated = n_phases * I_ph_rated^2 * R_ph_ref
        # I_ph_rated = sqrt(rated torque * omega_rated / n_phases / R_ph_ref * frac_copper)
        # Use fraction approach: iron + copper + other at rated point
        # At rated: P_iron_rated + P_mech_rated fixed; rest is copper
        omega_r = self.omega_rated
        P_iron_r = self.k_iron_eddy * omega_r**2 + self.k_iron_hyst * omega_r
        P_mech_r = self.k_mech * omega_r
        P_stray_r = self.k_stray * self.P_rated
        P_total_loss_target = self.P_rated * 0.10  # assume ~10% losses at rated (90% eff)
        P_copper_r = max(P_total_loss_target - P_iron_r - P_mech_r - P_stray_r, P_total_loss_target * 0.1)
        self._I_ph_rated = np.sqrt(P_copper_r / (self.n_phases * self.R_ph_ref))

    def phase_resistance(self, winding_temperature):
        """R_ph(T) = R_ph_ref * (1 + alpha_Cu*(T - T_ref))."""
        T_w = np.asarray(winding_temperature, dtype=float)
        return self.R_ph_ref * (1.0 + self.alpha_Cu * (T_w - self.T_ref))

    def phase_current(self, torque_nm, speed_rpm):
        """
        Approximate phase current [A].
        Scales torque demand relative to rated: I ~ I_rated * sqrt(T/T_rated).
        Uses sqrt because P_copper ~ I^2 and T ~ I at first approx for SRM.
        """
        T = np.asarray(torque_nm, dtype=float)
        T_ratio = np.where(self.T_rated > 0, np.abs(T) / self.T_rated, 0.0)
        return self._I_ph_rated * np.sqrt(np.maximum(T_ratio, 0.0))

    def iron_loss(self, speed_rpm):
        """Iron loss [W]: eddy (omega^2) + hysteresis (omega)."""
        omega = np.asarray(speed_rpm, dtype=float)
        return self.k_iron_eddy * omega**2 + self.k_iron_hyst * np.abs(omega)

    def mechanical_loss(self, speed_rpm):
        """Friction + windage [W]."""
        omega = np.asarray(speed_rpm, dtype=float)
        return self.k_mech * np.abs(omega)

    def output_power(self, torque_nm, speed_rpm):
        """Shaft power [W]."""
        T = np.asarray(torque_nm, dtype=float)
        omega = np.asarray(speed_rpm, dtype=float) * _RPM_TO_RADS
        return T * omega

    def losses(self, torque_nm, speed_rpm, winding_temperature=75.0):
        """
        Full loss breakdown [W].

        Returns dict: p_copper_w, p_iron_w, p_mech_w, p_stray_w, p_total_w.
        """
        T = np.asarray(torque_nm, dtype=float)
        omega = np.asarray(speed_rpm, dtype=float)

        R_ph = self.phase_resistance(winding_temperature)
        I_ph = self.phase_current(T, omega)
        P_copper = self.n_phases * I_ph**2 * R_ph

        P_iron = self.iron_loss(omega)
        P_mech = self.mechanical_loss(omega)
        P_out = self.output_power(T, omega)
        P_stray = self.k_stray * np.abs(P_out)
        P_total = P_copper + P_iron + P_mech + P_stray

        return {
            "p_copper_w": P_copper,
            "p_iron_w": P_iron,
            "p_mech_w": P_mech,
            "p_stray_w": P_stray,
            "p_total_w": P_total,
        }

    def input_power(self, torque_nm, speed_rpm, winding_temperature=75.0):
        """Electrical input power [W]."""
        P_out = self.output_power(torque_nm, speed_rpm)
        loss = self.losses(torque_nm, speed_rpm, winding_temperature)
        return P_out + loss["p_total_w"]

    def efficiency(self, torque_nm, speed_rpm, winding_temperature=75.0):
        """Efficiency = P_out / P_in."""
        P_out = self.output_power(torque_nm, speed_rpm)
        P_in = self.input_power(torque_nm, speed_rpm, winding_temperature)
        eta = np.where(P_in > 1e-6, P_out / P_in, 0.0)
        return np.clip(eta, 0.0, 1.0)

    def derating_factor(self, ambient_temperature):
        """IEC 60034-1 ambient derating above 40 C."""
        T_a = np.asarray(ambient_temperature, dtype=float)
        d = 1.0 - self.derating_slope * np.maximum(T_a - self.ambient_threshold, 0.0)
        return np.clip(d, 0.0, 1.0)
