"""
EC053 — Thermophotovoltaic (TPV) — F1a Efficiency Model

P = eta * epsilon * sigma * T_emitter^4 * A_cell * F_view

where:
    eta = cell efficiency at converting in-band radiation
    epsilon = emitter emissivity
    sigma = Stefan-Boltzmann constant
    T_emitter = emitter temperature (K)
    F_view = view factor from emitter to cell

References:
    Coutts (1999). A review of progress in thermophotovoltaic generation.
        Renewable and Sustainable Energy Reviews.
    Bauer (2011). Thermophotovoltaics. Springer.
"""

import numpy as np

SIGMA = 5.67e-8  # W/m²/K⁴


class TPVF1a:
    """Simple efficiency model for TPV cell."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta = u["eta"]["value"]
        self.T_emitter_ref = u["T_emitter"]["value"]
        self.A_cell = u["A_cell"]["value"]
        self.F_view = u["F_view"]["value"]
        self.emissivity = u["emissivity"]["value"]

    def predict(self, T_emitter=None, F_view=None):
        T = np.asarray(T_emitter if T_emitter is not None else self.T_emitter_ref, dtype=float)
        Fv = np.asarray(F_view if F_view is not None else self.F_view, dtype=float)

        T = np.maximum(T, 0.0)
        irradiance = self.emissivity * SIGMA * T ** 4  # W/m²
        P_incident = irradiance * self.A_cell * Fv    # W
        P_elec = self.eta * P_incident                # W

        return {
            "P_W": P_elec,
            "P_incident_W": P_incident,
            "irradiance_Wm2": irradiance,
            "eta_sys": self.eta,
        }
