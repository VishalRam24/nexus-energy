"""
EC105 — Gas Turbine CHP — F1a Heat-to-Power Ratio Model

Combined heat & power from a gas turbine + Heat Recovery Steam Generator (HRSG).

  eta_el(PLR) = eta_el_rated * (b0 + b1*PLR + b2*PLR^2)
  eta_th(PLR) = eta_th_rated * (c0 + c1*PLR)
  P_el        = P_el_rated * PLR                       [kW_e]
  fuel        = P_el / eta_el                          [kW_fuel, LHV]
  Q_th        = fuel * eta_th                          [kW_th]
  HPR         = Q_th / P_el                            [heat-to-power ratio]
  eta_total   = eta_el + eta_th                        [~0.80 typical]

References:
    US EPA CHP Catalog (2017). Combined Heat and Power Technology Fact Sheets.
    Kehlhofer et al. (2009). Combined-Cycle Gas & Steam Turbine Power Plants, 3rd ed.
"""

import numpy as np


class GasTurbineCHPF1a:
    """Gas turbine CHP heat-to-power ratio model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_el_rated = u["P_el_rated"]["value"]     # kW_e
        self.eta_el_rated = u["eta_el_rated"]["value"]
        self.eta_th_rated = u["eta_th_rated"]["value"]
        self.PLR_min = u["PLR_min"]["value"]
        self.b0 = u["b0"]["value"]
        self.b1 = u["b1"]["value"]
        self.b2 = u["b2"]["value"]
        self.c0 = u["c0"]["value"]
        self.c1 = u["c1"]["value"]

    def eta_electrical(self, plr):
        """Electrical efficiency as a function of PLR."""
        p = np.asarray(plr, dtype=float)
        eta = self.eta_el_rated * (self.b0 + self.b1 * p + self.b2 * p**2)
        return np.clip(eta, 0.0, 1.0)

    def eta_thermal(self, plr):
        """Thermal (HRSG) efficiency as a function of PLR."""
        p = np.asarray(plr, dtype=float)
        eta = self.eta_th_rated * (self.c0 + self.c1 * p)
        return np.clip(eta, 0.0, 1.0)

    def compute(self, part_load_ratio):
        """Compute power flows, efficiencies and HPR at given PLR.

        Parameters
        ----------
        part_load_ratio : float or array  [PLR_min – 1.0]

        Returns
        -------
        dict: electrical_power_kw, thermal_power_kw, fuel_input_kw,
              eta_electrical, eta_thermal, eta_total, heat_to_power_ratio
        """
        plr = np.asarray(part_load_ratio, dtype=float)
        plr = np.clip(plr, self.PLR_min, 1.0)

        eta_el = self.eta_electrical(plr)
        eta_th = self.eta_thermal(plr)

        P_el = self.P_el_rated * plr                              # kW_e
        fuel = np.where(eta_el > 1e-6, P_el / eta_el, 0.0)        # kW_fuel
        Q_th = fuel * eta_th                                      # kW_th
        eta_total = eta_el + eta_th
        hpr = np.where(P_el > 1e-6, Q_th / P_el, 0.0)

        return {
            "electrical_power_kw": P_el,
            "thermal_power_kw": Q_th,
            "fuel_input_kw": fuel,
            "eta_electrical": eta_el,
            "eta_thermal": eta_th,
            "eta_total": eta_total,
            "heat_to_power_ratio": hpr,
        }
