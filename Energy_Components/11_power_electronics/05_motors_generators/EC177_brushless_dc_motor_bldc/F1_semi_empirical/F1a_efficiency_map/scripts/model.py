"""
EC177 -- Brushless DC Motor (BLDC) -- F1a Efficiency Map

Efficiency model using two-component loss model:
    eta(PLR) = PLR / (PLR + c0 + c2 * PLR^2)

where:
    c0 = 0.35 * (1/eta_rated - 1)   [iron losses + windage, constant fraction]
    c2 = 0.65 * (1/eta_rated - 1)   [copper I^2R losses, load-dependent]

Torque-speed relationship:
    T_rated = P_rated / omega_rated
    T = PLR * T_rated
    omega = omega_rated   (speed-controlled by ESC/controller)

Current from torque constant:
    I = T / Kt

References:
    Hanselman, D.C. (2006). Brushless Permanent Magnet Motor Design. Magna Physics.
    Gieras, J.F. (2010). Permanent Magnet Motor Technology. CRC Press.
"""

import numpy as np


class BLDCF1a:
    """BLDC Motor -- efficiency map from rated efficiency + Kt-based torque model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["rated_power"]["value"]   # kW
        self.eta_rated = u["eta_rated"]["value"]
        self.omega_rated = u["omega_rated"]["value"]  # rpm
        self.Kt = u["Kt"]["value"]                 # Nm/A
        # Two-component loss model
        total_loss_frac = 1.0 / self.eta_rated - 1.0
        self.c0 = 0.35 * total_loss_frac
        self.c2 = 0.65 * total_loss_frac
        # Rated torque [Nm]
        omega_rad = self.omega_rated * 2.0 * np.pi / 60.0
        self.T_rated = (self.P_rated * 1000.0) / omega_rad  # Nm

    def efficiency(self, plr):
        """eta = PLR / (PLR + c0 + c2*PLR^2) -- peaks near PLR = sqrt(c0/c2)."""
        plr = np.asarray(plr, dtype=float)
        eta = plr / (plr + self.c0 + self.c2 * plr**2)
        return np.clip(eta, 1e-6, 0.9999)

    def torque(self, plr):
        """Output torque [Nm] = PLR * T_rated."""
        return np.asarray(plr, dtype=float) * self.T_rated

    def current(self, plr):
        """Motor current [A] from torque constant: I = T / Kt."""
        return self.torque(plr) / self.Kt

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
