"""
EC067 — Airborne Wind Energy (AWE) — F1a Power Curve Model

Simplified crosswind kite power model:
    P = 0.5 * rho * Cp_eff * A_wing * v^3 * eta_gen  (v_in <= v <= v_rated)
    P = P_rated                                        (v_rated < v <= v_out)
    P = 0                                              (otherwise)

Loyd (1980) maximum crosswind power: P_max = (4/27) * rho * A * v^3 * (CL^3/CD^2)
Here simplified to Cp_eff parameter.

References:
    Loyd (1980). Crosswind kite power. J. Energy, 4(3), 106-111.
    Diehl (2013). Airborne Wind Energy. Springer.
    Fagiano & Milanese (2012). Airborne wind energy: Basic concepts and physical foundations.
"""

import numpy as np


class AWEF1a:
    """Power curve model for airborne wind energy system."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.A = u["A_wing"]["value"]
        self.CL_CD = u["CL_CD"]["value"]
        self.P_rated = u["P_rated"]["value"]
        self.v_in = u["v_cut_in"]["value"]
        self.v_rated = u["v_rated"]["value"]
        self.v_out = u["v_cut_out"]["value"]
        self.rho = u["rho"]["value"]
        self.eta = u["eta_gen"]["value"]
        self.Cp_eff = u["Cp_eff"]["value"]

    def predict(self, v, rho=None):
        v = np.asarray(v, dtype=float)
        rho = np.asarray(rho if rho is not None else self.rho, dtype=float)

        P_aero = 0.5 * rho * self.Cp_eff * self.A * v ** 3 * self.eta
        P_aero = np.minimum(P_aero, self.P_rated)

        P_W = np.where(v < self.v_in, 0.0,
               np.where(v > self.v_out, 0.0, P_aero))

        return {
            "P_kW": P_W / 1000.0,
            "P_rated_kW": self.P_rated / 1000.0,
            "capacity_factor": P_W / self.P_rated,
        }
