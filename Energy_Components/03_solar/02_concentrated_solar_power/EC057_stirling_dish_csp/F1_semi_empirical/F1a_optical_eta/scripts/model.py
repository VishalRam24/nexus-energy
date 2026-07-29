"""
EC057 — Stirling Dish CSP — F1a Optical Efficiency Model

P = eta_opt * eta_engine * DNI * A_dish * cos(theta)

Stirling dish achieves highest concentration ratio (~3000x) and efficiency
among CSP technologies (30-32% net electric). Two-axis tracking.

References:
    Mancini et al. (2003). Dish-Stirling systems: An overview of development
        and status. ASME J. Sol. Energy Eng.
    Stine & Diver (1994). A Compendium of Solar Dish/Stirling Technology.
"""

import numpy as np


class StirlingDishF1a:
    """Optical + engine efficiency model for Stirling dish CSP."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta_opt = u["eta_opt"]["value"]
        self.eta_engine = u["eta_engine"]["value"]
        self.A_dish = u["A_dish"]["value"]

    def predict(self, DNI, theta=0.0):
        DNI = np.asarray(DNI, dtype=float)
        theta = np.asarray(theta, dtype=float)

        cos_theta = np.cos(np.radians(theta))
        Q_focal = self.eta_opt * DNI * self.A_dish * cos_theta
        Q_focal = np.maximum(Q_focal, 0.0)
        P_elec = self.eta_engine * Q_focal

        return {
            "P_kW": P_elec / 1000.0,
            "Q_focal_kW": Q_focal / 1000.0,
            "eta_optical": self.eta_opt * cos_theta,
            "eta_system": self.eta_opt * self.eta_engine * cos_theta,
        }
