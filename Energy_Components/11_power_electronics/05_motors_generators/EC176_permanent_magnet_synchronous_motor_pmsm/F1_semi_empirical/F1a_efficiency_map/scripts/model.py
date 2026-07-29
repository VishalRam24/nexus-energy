"""
EC176 — PMSM — F1a Efficiency Map Model

Loss separation approach:
    P_copper = (T / k_t)^2 * R_s          (copper/I2R loss)
    P_iron   = k_e * omega^1.5             (core loss, Steinmetz-like)
    P_mech   = k_f * omega                 (friction + windage)
    P_loss   = P_copper + P_iron + P_mech
    P_out    = T * omega
    P_in     = P_out + P_loss
    eta      = P_out / P_in

Reference:
    Gieras, J.F. (2010). Permanent Magnet Motor Technology, 3rd ed. CRC Press.
    Chapter 3: Losses and efficiency of PM motors.
"""

import numpy as np

_RPM_TO_RADS = np.pi / 30.0


class PMSMF1a:
    """PMSM efficiency map via loss separation (copper + iron + mechanical)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["P_rated"]["value"]        # W
        self.T_rated = u["T_rated"]["value"]        # Nm
        self.omega_base = u["omega_base"]["value"]  # rpm
        self.omega_max = u["omega_max"]["value"]    # rpm
        self.eta_peak = u["eta_peak"]["value"]
        self.k_t = u["k_t"]["value"]               # Nm/A  (torque constant)
        self.R_s = u["R_s"]["value"]               # ohm
        self.k_e = u["k_e"]["value"]               # iron loss coeff
        self.k_f = u["k_f"]["value"]               # mech loss coeff

    def losses(self, torque_nm, speed_rpm):
        """
        Compute individual and total losses.

        Args:
            torque_nm:  Output torque in Nm.
            speed_rpm:  Speed in rpm.

        Returns:
            dict with p_copper_w, p_iron_w, p_mech_w, p_total_w (all in W).
        """
        T = np.asarray(torque_nm, dtype=float)
        omega = np.asarray(speed_rpm, dtype=float)  # rpm for loss coefficients

        # Copper loss: P = I^2 * R, I = T / k_t (3-phase, simplified to equivalent)
        I = T / self.k_t
        P_copper = I ** 2 * self.R_s

        # Iron loss: Steinmetz-like, proportional to omega^1.5
        P_iron = self.k_e * np.abs(omega) ** 1.5

        # Mechanical loss: friction + windage ∝ omega
        P_mech = self.k_f * np.abs(omega)

        P_total = P_copper + P_iron + P_mech

        return {
            "p_copper_w": P_copper,
            "p_iron_w": P_iron,
            "p_mech_w": P_mech,
            "p_total_w": P_total,
        }

    def output_power(self, torque_nm, speed_rpm):
        """Mechanical output power in W: P_out = T * omega_rad."""
        T = np.asarray(torque_nm, dtype=float)
        omega_rad = np.asarray(speed_rpm, dtype=float) * _RPM_TO_RADS
        return T * omega_rad

    def input_power(self, torque_nm, speed_rpm):
        """Electrical input power in W: P_in = P_out + P_loss."""
        p_out = self.output_power(torque_nm, speed_rpm)
        loss = self.losses(torque_nm, speed_rpm)
        return p_out + loss["p_total_w"]

    def efficiency(self, torque_nm, speed_rpm):
        """
        Motor efficiency: eta = P_out / P_in.
        Returns 0 when T=0 or omega=0 (no output power).
        """
        T = np.asarray(torque_nm, dtype=float)
        omega = np.asarray(speed_rpm, dtype=float)
        p_out = self.output_power(T, omega)
        p_in = self.input_power(T, omega)
        # Avoid division by zero: if p_in is very small, eta = 0
        eta = np.where(p_in > 1e-6, p_out / p_in, 0.0)
        return np.clip(eta, 0.0, 1.0)
