"""
EC168 — MPPT Controller — F1a Tracking Efficiency Model

eta_mppt = eta_max * (1 - exp(-k * G / G_ref))
P_out = P_mpp * eta_mppt
power_loss = P_mpp - P_out

At G > 200 W/m2: eta ~ 0.98-0.99
At G < 50 W/m2:  eta drops to 0.90-0.92

Reference:
    Hohm, D.P. & Ropp, M.E. (2003). Comparative study of maximum power point
    tracking algorithms. Progress in Photovoltaics, 11, 47-62.
"""

import numpy as np


class MPPTF1a:
    """MPPT controller — tracking efficiency as a function of irradiance."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta_max = u["eta_max"]["value"]
        self.k = u["k"]["value"]
        self.G_ref = u["G_ref"]["value"]
        self.P_rated = u["P_rated"]["value"]

    def tracking_efficiency(self, irradiance):
        """
        MPPT tracking efficiency at given irradiance.

        Args:
            irradiance: Solar irradiance in W/m2.

        Returns:
            eta_mppt (dimensionless, 0 <= eta <= eta_max).
        """
        G = np.asarray(irradiance, dtype=float)
        eta = self.eta_max * (1.0 - np.exp(-self.k * G / self.G_ref))
        # At exactly G=0: eta=0 (no power to track)
        eta = np.where(G <= 0.0, 0.0, eta)
        return np.clip(eta, 0.0, 1.0)

    def output_power(self, irradiance, p_mpp_input):
        """
        MPPT output power in W.

        Args:
            irradiance:   Solar irradiance in W/m2.
            p_mpp_input:  Available MPP power from PV array in W.

        Returns:
            P_out in W.
        """
        eta = self.tracking_efficiency(irradiance)
        p_in = np.asarray(p_mpp_input, dtype=float)
        return p_in * eta

    def power_loss(self, irradiance, p_mpp_input):
        """Tracking loss in W (P_mpp - P_out)."""
        p_in = np.asarray(p_mpp_input, dtype=float)
        p_out = self.output_power(irradiance, p_mpp_input)
        return p_in - p_out
