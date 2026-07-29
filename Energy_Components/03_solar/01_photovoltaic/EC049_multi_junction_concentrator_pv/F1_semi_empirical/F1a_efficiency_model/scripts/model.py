"""
EC049 — Multi-Junction Concentrator PV — F1a Efficiency Model

Simplified CPV power model:
    P = eta_eff * DNI * C_ratio * A_cell * cos(theta)
    eta_eff = eta_ref * (1 + beta_T * (T_cell - T_ref))

References:
    Luque & Hegedus (2011). Handbook of Photovoltaic Science and Engineering.
    Kinsey et al. (2008). Concentrator multifunction solar cell characteristics.
"""

import numpy as np


class MultiJunctionCPVF1a:
    """Simplified efficiency model for III-V multi-junction CPV."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta_ref = u["eta"]["value"]
        self.C_ratio = u["C_ratio"]["value"]
        self.A_cell = u["A_cell"]["value"]
        self.T_ref = u["T_ref"]["value"]
        self.beta_T = u["beta_T"]["value"]

    def predict(self, DNI, T_cell=25.0, theta_incidence=0.0):
        DNI = np.asarray(DNI, dtype=float)
        T_cell = np.asarray(T_cell, dtype=float)
        theta = np.asarray(theta_incidence, dtype=float)

        theta_rad = np.radians(theta)
        cos_theta = np.cos(theta_rad)

        eta_eff = self.eta_ref * (1.0 + self.beta_T * (T_cell - self.T_ref))
        eta_eff = np.clip(eta_eff, 0.0, 1.0)

        P_W = eta_eff * DNI * self.C_ratio * self.A_cell * cos_theta
        P_W = np.maximum(P_W, 0.0)

        irr_on_cell = DNI * self.C_ratio * cos_theta
        P_max_theoretical = self.eta_ref * irr_on_cell * self.A_cell

        return {
            "P_W": P_W,
            "eta_eff": eta_eff,
            "irr_concentrated": irr_on_cell,
            "P_max_ref": P_max_theoretical,
        }
