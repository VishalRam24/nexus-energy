"""
EC175 — Induction Motor/Generator — F1a Efficiency Map

Efficiency model per IEC 60034-30-1 using two-component loss model:
    P_loss = P_rated * (c0 + c2 * PLR^2)
    eta(PLR) = PLR / (PLR + c0 + c2 * PLR^2)

where:
    c0  = constant losses fraction (iron core, friction, windage) / P_rated
    c2  = variable losses fraction (copper I²R) at PLR=1 / P_rated

Coefficients are derived from rated efficiency:
    c0 + c2 = 1/eta_rated - 1
    c0 = 0.35 * (1/eta_rated - 1)   [~35% constant, ~65% variable — typical IE3]
    c2 = 0.65 * (1/eta_rated - 1)

This formulation correctly produces peak efficiency near PLR = sqrt(c0/c2) ≈ 0.73–0.80,
consistent with IEC 60034-30-1 measurements.

Slip:
    s = (omega_sync - omega_rotor) / omega_sync
    At full load: s_rated

References:
    IEC 60034-30-1:2014 (IE efficiency classes)
    Boldea, I. & Nasar, S.A. (2010). The Induction Machine Handbook. CRC Press.
"""

import numpy as np


class InductionMotorF1a:
    """Induction motor/generator — efficiency map from IEC 60034-30-1."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["rated_power"]["value"]        # kW
        self.eta_rated = u["eta_rated"]["value"]
        self.pf = u["power_factor"]["value"]
        self.n_sync = u["sync_speed"]["value"]          # rpm
        self.n_rated = u["rated_speed"]["value"]        # rpm
        self.s_rated = u["rated_slip"]["value"]
        # Two-component loss model calibrated to rated efficiency
        total_loss_frac = 1.0 / self.eta_rated - 1.0  # = (1 - eta_rated) / eta_rated
        self.c0 = 0.35 * total_loss_frac   # constant losses / P_rated
        self.c2 = 0.65 * total_loss_frac   # variable (copper) losses / P_rated at PLR=1

    def efficiency(self, plr):
        """
        Efficiency as function of part-load ratio [0 < PLR <= 1.2].
        Uses IEC 60034-30-1 two-component loss model:
            eta = PLR / (PLR + c0 + c2*PLR^2)
        Peaks at PLR = sqrt(c0/c2).
        """
        plr = np.asarray(plr, dtype=float)
        eta = plr / (plr + self.c0 + self.c2 * plr ** 2)
        # Physical limits
        return np.clip(eta, 1e-6, 0.9999)

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
        """Total losses [kW] = P_in - P_out."""
        return self.input_power(plr) - self.output_power(plr)

    def slip(self, plr=1.0):
        """
        Approximate slip as function of load.
        For induction motor: s ~ s_rated * PLR (linear approximation for low slip).
        """
        plr = np.asarray(plr, dtype=float)
        return np.clip(self.s_rated * plr, 0.0, 1.0)

    def rotor_speed(self, plr=1.0):
        """Rotor speed [rpm]."""
        s = self.slip(plr)
        return self.n_sync * (1.0 - s)
