"""
EC180 — Doubly-Fed Induction Generator (DFIG) — F1a Efficiency Map

Simplified efficiency map model for DFIG wind generator.
Variable speed operation: slip range ±30% around synchronous speed.

Model:
    eta(PLR, slip) = eta_rated * f_partload(PLR) * f_slip(slip)

    f_partload(PLR) = 1 - eta_partload_coeff * max(1 - PLR, 0)
        (efficiency drops slightly below rated load)

    f_slip(slip) = 1 - 0.5 * |slip|^2
        (small penalty for operating far from synchronous, due to converter losses)

    P_out = PLR * P_rated
    P_in = P_out / eta

References:
    Muller, S. et al. (2002). Doubly Fed Induction Generator Systems for Wind Turbines.
      IEEE Industry Applications Magazine 8(3):26-33.
    IEC 61400-21:2008.
"""

import numpy as np


class DFIGF1a:
    """Simple DFIG efficiency map model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["P_rated_MW"]["value"] * 1e6  # W
        self.eta_rated = u["eta_rated"]["value"]
        self.slip_min = u["slip_min"]["value"]
        self.slip_max = u["slip_max"]["value"]
        self.omega_sync = u["omega_sync_rpm"]["value"]
        self.eta_partload_coeff = u["eta_partload_coeff"]["value"]

    def efficiency(self, load_fraction, slip=0.0):
        """Overall DFIG efficiency [-]."""
        plr = np.asarray(load_fraction, dtype=float)
        s = np.asarray(slip, dtype=float)
        # Part-load correction: drops linearly below rated
        f_pl = 1.0 - self.eta_partload_coeff * np.maximum(1.0 - plr, 0.0)
        # Slip correction: small converter penalty away from synchronous
        f_slip = 1.0 - 0.5 * s**2
        eta = self.eta_rated * f_pl * f_slip
        return np.clip(eta, 0.0, 1.0)

    def output_power(self, load_fraction):
        """Electrical output power [W]."""
        return np.asarray(load_fraction, dtype=float) * self.P_rated

    def input_power(self, load_fraction, slip=0.0):
        """Mechanical shaft input power [W]."""
        P_out = self.output_power(load_fraction)
        eta = self.efficiency(load_fraction, slip)
        return np.where(eta > 0.0, P_out / eta, 0.0)

    def rotor_speed_rpm(self, slip):
        """Rotor speed [rpm] = omega_sync * (1 - slip)."""
        s = np.asarray(slip, dtype=float)
        return self.omega_sync * (1.0 - s)
