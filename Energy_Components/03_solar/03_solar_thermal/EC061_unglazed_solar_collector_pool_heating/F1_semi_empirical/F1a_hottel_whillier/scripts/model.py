"""
EC061 — Unglazed Solar Collector (Pool Heating) — F1a Hottel-Whillier

Q = A * [eta0 * G - a1 * (Tm - Ta)]

Unglazed collectors have:
- High eta0 (~0.90) — no reflection from glass cover
- High a1 (~15 W/m²K) — large convective + radiative losses
- Optimal for low-temperature applications (Tm ≈ Ta + 5-15K)

References:
    Duffie & Beckman (2013). Solar Engineering of Thermal Processes, ch.6.
    ISO 9806:2017. Solar energy — Solar thermal collectors — Test methods.
"""

import numpy as np


class UnglazedCollectorF1a:
    """Hottel-Whillier model for unglazed solar collector."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta0 = u["eta0"]["value"]
        self.a1 = u["a1"]["value"]
        self.a2 = u["a2"]["value"]
        self.A = u["A"]["value"]

    def predict(self, G, Tm=28.0, Ta=20.0):
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
            "Q_W": Q_W,
            "eta": eta,
            "dT": dT,
        }
