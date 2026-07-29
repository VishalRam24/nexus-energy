"""
EC064 — Small/Micro Wind Turbine — F1a Power Curve Model

P = 0.5 * rho * Cp * A * v^3 * eta_gen   (between v_cut_in and v_rated)
P = P_rated                               (between v_rated and v_cut_out)
P = 0                                     (below v_cut_in or above v_cut_out)

where A = pi/4 * D^2

References:
    Manwell et al. (2009). Wind Energy Explained: Theory, Design and Application. 2nd ed.
    IEC 61400-2:2013. Small wind turbines.
"""

import numpy as np


class SmallWindTurbineF1a:
    """Power curve model for small/micro wind turbine."""

    def __init__(self, params: dict):
        u = params["unit"]
        D = u["D"]["value"]
        self.A = np.pi / 4.0 * D ** 2
        self.P_rated = u["P_rated"]["value"]
        self.v_in = u["v_cut_in"]["value"]
        self.v_rated = u["v_rated"]["value"]
        self.v_out = u["v_cut_out"]["value"]
        self.Cp = u["Cp"]["value"]
        self.rho = u["rho"]["value"]
        self.eta_gen = u["eta_gen"]["value"]

    def predict(self, v, rho=None):
        v = np.asarray(v, dtype=float)
        rho = np.asarray(rho if rho is not None else self.rho, dtype=float)

        P_aero = 0.5 * rho * self.Cp * self.A * v ** 3 * self.eta_gen
        P_aero = np.minimum(P_aero, self.P_rated)

        P_W = np.where(v < self.v_in, 0.0,
               np.where(v > self.v_out, 0.0, P_aero))

        Cp_actual = np.where(
            (v >= self.v_in) & (v <= self.v_out) & (v > 0),
            P_W / np.maximum(0.5 * rho * self.A * v ** 3, 1e-10),
            0.0
        )

        return {
            "P_kW": P_W / 1000.0,
            "Cp_actual": Cp_actual,
            "P_aero_kW": np.minimum(P_aero, self.P_rated) / 1000.0,
        }
