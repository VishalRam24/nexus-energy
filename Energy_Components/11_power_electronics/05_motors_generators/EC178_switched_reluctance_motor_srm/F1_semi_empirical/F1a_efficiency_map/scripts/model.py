"""
EC178 -- Switched Reluctance Motor (SRM) -- F1a Efficiency Map

SRM efficiency using two-component loss model:
    eta(PLR) = PLR / (PLR + c0 + c2 * PLR^2)

SRM characteristics:
    - Higher iron losses than PMSM (hence lower eta_peak = 0.88)
    - High torque ripple (~15% peak-to-peak)
    - No permanent magnets (robust, fault-tolerant)
    - Doubly salient structure (12/8 poles typical)

Torque with ripple estimate:
    T_avg = PLR * T_rated
    T_ripple = torque_ripple_factor * T_avg   [peak-to-peak estimate]

References:
    Krishnan, R. (2001). Switched Reluctance Motor Drives: Modeling, Simulation, Analysis,
    Design, and Applications. CRC Press.
"""

import numpy as np


class SRMF1a:
    """Switched Reluctance Motor -- efficiency map + torque ripple estimate."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["rated_power"]["value"]    # kW
        self.eta_rated = u["eta_rated"]["value"]
        self.omega_rated = u["omega_rated"]["value"]  # rpm
        self.ripple_factor = u["torque_ripple_factor"]["value"]
        # Two-component loss model
        total_loss_frac = 1.0 / self.eta_rated - 1.0
        self.c0 = 0.35 * total_loss_frac
        self.c2 = 0.65 * total_loss_frac
        # Rated torque [Nm]
        omega_rad = self.omega_rated * 2.0 * np.pi / 60.0
        self.T_rated = (self.P_rated * 1000.0) / omega_rad  # Nm

    def efficiency(self, plr):
        """eta = PLR / (PLR + c0 + c2*PLR^2)."""
        plr = np.asarray(plr, dtype=float)
        eta = plr / (plr + self.c0 + self.c2 * plr**2)
        return np.clip(eta, 1e-6, 0.9999)

    def torque_avg(self, plr):
        """Average output torque [Nm]."""
        return np.asarray(plr, dtype=float) * self.T_rated

    def torque_ripple(self, plr):
        """Peak-to-peak torque ripple estimate [Nm]."""
        return self.ripple_factor * self.torque_avg(plr)

    def output_power(self, plr):
        """Mechanical output power [kW]."""
        return np.asarray(plr, dtype=float) * self.P_rated

    def input_power(self, plr):
        """Electrical input power [kW]."""
        plr = np.asarray(plr, dtype=float)
        eta = self.efficiency(plr)
        p_out = self.output_power(plr)
        return np.where(eta > 0, p_out / eta, 0.0)

    def losses(self, plr):
        """Total losses [kW]."""
        return self.input_power(plr) - self.output_power(plr)
