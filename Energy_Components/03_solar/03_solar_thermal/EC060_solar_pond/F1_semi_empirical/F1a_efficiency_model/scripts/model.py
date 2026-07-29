"""
EC060 — Solar Pond — F1a Efficiency Model (Hottel-Whillier)

Q = A * [eta0*G - a1*(Tm - Ta) - a2*(Tm - Ta)^2]
eta = Q / (A * G)

Solar ponds have very low efficiency (~5%) due to large optical losses in
the brine layers, but can store large amounts of thermal energy cheaply.

References:
    Tabor (1981). Solar ponds. Solar Energy.
    Hull et al. (1988). The cost effectiveness of salt gradient solar ponds.
    Duffie & Beckman (2013). Solar Engineering of Thermal Processes, ch.9.
"""

import numpy as np


class SolarPondF1a:
    """Hottel-Whillier model for solar pond."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta0 = u["eta0"]["value"]
        self.a1 = u["a1"]["value"]
        self.a2 = u["a2"]["value"]
        self.A = u["A"]["value"]

    def predict(self, G, Tm=80.0, Ta=20.0):
        G = np.asarray(G, dtype=float)
        Tm = np.asarray(Tm, dtype=float)
        Ta = np.asarray(Ta, dtype=float)

        dT = Tm - Ta
        Q_W = self.A * (self.eta0 * G - self.a1 * dT - self.a2 * dT ** 2)
        Q_W = np.maximum(Q_W, 0.0)

        G_safe = np.where(G < 1e-6, 1.0, G)
        eta = np.where(G < 1e-6, 0.0, Q_W / (self.A * G_safe))
        eta = np.clip(eta, 0.0, 1.0)

        return {
            "Q_kW": Q_W / 1000.0,
            "eta": eta,
            "dT": dT,
        }
