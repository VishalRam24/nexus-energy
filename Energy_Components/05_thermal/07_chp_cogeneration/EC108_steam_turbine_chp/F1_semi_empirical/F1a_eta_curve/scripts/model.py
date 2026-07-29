"""
EC108 -- Steam Turbine CHP -- F1a Efficiency-Curve Model

Simplest semi-empirical model for a back-pressure steam turbine CHP unit.
Applies constant rated efficiencies scaled by part-load ratio (PLR):

    P_el  = P_el_rated * PLR * eta_el_rated   [kW_e]
    Q_th  = P_el_rated * PLR / eta_el_rated * eta_th_rated  [kW_th]

Wait -- cleaner formulation:
    P_fuel = P_el_rated * PLR / eta_el_rated   (fuel input)
    P_el   = eta_el_rated * P_fuel             (electrical output)
    Q_th   = eta_th_rated * P_fuel             (thermal output)
    eta_total = eta_el_rated + eta_th_rated

This is the F1a "constant-efficiency" baseline. F1b adds quadratic PLR
curves and ambient temperature derating.

References:
    US EPA CHP Catalog (2017). Combined Heat and Power Technology Fact Sheets.
    IEA (2008). Combined Heat and Power.
"""

import numpy as np


class SteamTurbineCHPF1a:
    """Back-pressure steam turbine CHP: constant efficiency curve model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_el_rated   = float(u["P_el_rated_kw"]["value"])   # kW_e
        self.eta_el       = float(u["eta_el_rated"]["value"])
        self.eta_th       = float(u["eta_th_rated"]["value"])
        self.PLR_min      = float(u["PLR_min"]["value"])
        self.PLR_max      = float(u["PLR_max"]["value"])

    def predict(self, PLR: float) -> dict:
        """
        Compute CHP outputs for a given part-load ratio.

        Parameters
        ----------
        PLR : float
            Part-load ratio [0.3 – 1.0]

        Returns
        -------
        dict with:
            PLR_clamped      : effective PLR after clamping
            P_el_kw          : electrical output [kW_e]
            Q_th_kw          : thermal (steam) output [kW_th]
            P_fuel_kw        : fuel input [kW]
            eta_el           : electrical efficiency [-]
            eta_th           : thermal efficiency [-]
            eta_total        : combined efficiency [-]
            HPR              : heat-to-power ratio [-]
        """
        PLR = float(np.clip(PLR, self.PLR_min, self.PLR_max))

        P_fuel = self.P_el_rated * PLR / self.eta_el   # kW fuel input
        P_el   = self.eta_el * P_fuel                   # kW_e
        Q_th   = self.eta_th * P_fuel                   # kW_th
        eta_total = self.eta_el + self.eta_th
        HPR    = Q_th / P_el if P_el > 0 else 0.0

        return {
            "PLR_clamped": float(PLR),
            "P_el_kw":     float(P_el),
            "Q_th_kw":     float(Q_th),
            "P_fuel_kw":   float(P_fuel),
            "eta_el":      float(self.eta_el),
            "eta_th":      float(self.eta_th),
            "eta_total":   float(eta_total),
            "HPR":         float(HPR),
        }
