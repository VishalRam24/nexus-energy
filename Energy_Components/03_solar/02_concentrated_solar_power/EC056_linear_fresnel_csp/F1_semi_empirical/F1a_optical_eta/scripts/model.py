"""
EC056 — Linear Fresnel CSP — F1a Optical Efficiency Model

Q_thermal = eta_opt * eta_thermal * DNI * A_aperture * cos(theta)
P_electric = Q_thermal * eta_PB  (optional, power block)

Linear Fresnel has lower optical efficiency than parabolic trough (~60%)
but lower capital cost. Concentration ratio ~30-80x.

References:
    Morin et al. (2012). Comparison of Linear Fresnel and Parabolic Trough
        Collector power plants. Solar Energy.
    IEA-ETSAP (2010). Concentrating Solar Power.
"""

import numpy as np


class LinearFresnelF1a:
    """Optical efficiency model for linear Fresnel CSP."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta_opt = u["eta_opt"]["value"]
        self.A = u["A_aperture"]["value"]
        self.eta_th = u["eta_thermal"]["value"]

    def predict(self, DNI, theta=0.0, eta_PB=0.33):
        DNI = np.asarray(DNI, dtype=float)
        theta = np.asarray(theta, dtype=float)
        eta_PB = float(eta_PB)

        cos_theta = np.cos(np.radians(theta))
        Q_W = self.eta_opt * self.eta_th * DNI * self.A * cos_theta
        Q_W = np.maximum(Q_W, 0.0)
        P_W = Q_W * eta_PB

        return {
            "Q_MW": Q_W / 1e6,
            "P_MW": P_W / 1e6,
            "eta_optical": np.full_like(DNI, self.eta_opt) * cos_theta,
            "eta_system": np.full_like(DNI, self.eta_opt * self.eta_th * eta_PB) * cos_theta,
        }
