"""
EC104 — Gas Engine CHP — F1a Part-Load Efficiency Model

eta_el(PLR) and eta_th(PLR) — electrical and thermal efficiency vs part-load ratio:
  eta_el = eta_el_rated * (b0 + b1*PLR + b2*PLR^2)
  eta_th = eta_th_rated * (c0 + c1*PLR)
  fuel   = P_el_rated * PLR / eta_el           [kW_fuel]
  P_el   = fuel * eta_el = P_el_rated * PLR    [kW_e]
  Q_th   = fuel * eta_th                       [kW_th]
  eta_total = eta_el + eta_th                  [should be 0.80-0.90]

References:
    US EPA CHP Catalog (2017). Combined Heat and Power Technology Fact Sheets.
    ASUE BHKW-Kenndaten (2011). Blocks Heizkraftwerke — Kenndaten.
"""

import numpy as np


class GasEngineCHPF1a:
    """Gas engine CHP part-load efficiency model."""

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
        """Thermal efficiency as a function of PLR."""
        p = np.asarray(plr, dtype=float)
        eta = self.eta_th_rated * (self.c0 + self.c1 * p)
        return np.clip(eta, 0.0, 1.0)

    def compute(self, part_load_ratio):
        """Compute power flows and efficiencies at given PLR.

        Parameters
        ----------
        part_load_ratio : float or array  [PLR_min – 1.0]

        Returns
        -------
        dict: electrical_power_kw, thermal_power_kw, fuel_input_kw,
              eta_electrical, eta_thermal, eta_total
        """
        plr = np.asarray(part_load_ratio, dtype=float)
        plr = np.clip(plr, self.PLR_min, 1.0)

        eta_el = self.eta_electrical(plr)
        eta_th = self.eta_thermal(plr)

        # Fuel input: engine delivers P_el = P_rated * PLR electrical output
        # fuel = P_el / eta_el
        P_el = self.P_el_rated * plr                    # kW_e
        fuel = np.where(eta_el > 1e-6, P_el / eta_el, 0.0)   # kW_fuel
        Q_th = fuel * eta_th                            # kW_th
        eta_total = eta_el + eta_th

        return {
            "electrical_power_kw": P_el,
            "thermal_power_kw": Q_th,
            "fuel_input_kw": fuel,
            "eta_electrical": eta_el,
            "eta_thermal": eta_th,
            "eta_total": eta_total,
        }
