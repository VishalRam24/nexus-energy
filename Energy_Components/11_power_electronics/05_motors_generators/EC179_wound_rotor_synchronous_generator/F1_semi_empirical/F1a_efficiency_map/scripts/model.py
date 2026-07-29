"""
EC179 -- Wound Rotor Synchronous Generator -- F1a Efficiency Map

Generator efficiency via two-component loss model:
    eta(PLR) = PLR / (PLR + c0 + c2 * PLR^2)

Note: For a generator, PLR represents the ratio of actual electrical output
power to rated power. P_mech_in = P_elec_out / eta.

Generator-specific features:
    - Fixed synchronous speed: omega_s = 4*pi*f / poles
    - Variable excitation controls terminal voltage and reactive power
    - P_mech = P_elec + P_losses   (input from turbine/prime mover)

The F1a model uses the same two-loss formula as motors but represents
generator operation (mechanical input -> electrical output).

References:
    Boldea, I. (2015). Synchronous Generators, 2nd ed. CRC Press.
    Chapman, S.J. (2011). Electric Machinery Fundamentals, 5th ed. McGraw-Hill.
"""

import numpy as np


class WRSyncGenF1a:
    """Wound Rotor Synchronous Generator -- efficiency map (generator convention)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["rated_power"]["value"]   # kW
        self.eta_rated = u["eta_rated"]["value"]
        self.omega_rated = u["omega_rated"]["value"]  # rpm
        self.V_terminal = u["v_terminal"]["value"]   # V
        self.pf_rated = u["power_factor"]["value"]
        self.f = u["frequency"]["value"]
        self.poles = u["poles"]["value"]
        # Two-component loss model
        total_loss_frac = 1.0 / self.eta_rated - 1.0
        self.c0 = 0.35 * total_loss_frac
        self.c2 = 0.65 * total_loss_frac
        # Rated torque [kNm]
        omega_rad = self.omega_rated * 2.0 * np.pi / 60.0
        self.T_rated = (self.P_rated * 1000.0) / omega_rad  # Nm

    def efficiency(self, plr):
        """eta = PLR / (PLR + c0 + c2*PLR^2)."""
        plr = np.asarray(plr, dtype=float)
        eta = plr / (plr + self.c0 + self.c2 * plr**2)
        return np.clip(eta, 1e-6, 0.9999)

    def electrical_output(self, plr):
        """Electrical output power [kW] = PLR * P_rated."""
        return np.asarray(plr, dtype=float) * self.P_rated

    def mechanical_input(self, plr):
        """Mechanical input power [kW] from prime mover = P_elec / eta."""
        plr = np.asarray(plr, dtype=float)
        eta = self.efficiency(plr)
        p_elec = self.electrical_output(plr)
        return np.where(eta > 0, p_elec / eta, 0.0)

    def losses(self, plr):
        """Total losses [kW] = P_mech - P_elec."""
        return self.mechanical_input(plr) - self.electrical_output(plr)

    def terminal_current(self, plr, power_factor=None):
        """
        Terminal current [kA] (3-phase):
            I = P_elec / (sqrt(3) * V_LL * PF)
        """
        if power_factor is None:
            power_factor = self.pf_rated
        p_elec = self.electrical_output(plr) * 1000.0  # W
        pf = np.asarray(power_factor, dtype=float)
        denom = np.sqrt(3.0) * self.V_terminal * pf
        return np.where(denom > 0, p_elec / denom, 0.0)  # A

    def synchronous_speed_rpm(self):
        """Synchronous speed [rpm] = 120*f / poles."""
        return 120.0 * self.f / self.poles
