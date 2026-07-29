"""Empirical solar-fraction table for EC090 Solar Water Heater Combi System.

f_solar tabulated against irradiance G (W/m2), np.interp lookup. Auxiliary
boiler covers the remaining demand at eta_boiler.
Source: EC090 F1a solar combi params; Duffie & Beckman (2013) Solar Engineering of Thermal Processes
"""
import numpy as np


class SolarFractionModel:
    def __init__(self, params):
        t = params["table"]
        self.G = np.asarray(t["G"], dtype=float)
        self.f = np.asarray(t["f_solar"], dtype=float)
        self.eta_boiler = float(params["eta_boiler"]["value"])
        self.q_demand = float(params["Q_demand_W"]["value"])

    def solar_fraction(self, G):
        return float(np.clip(np.interp(G, self.G, self.f), 0.0, 1.0))

    def predict(self, inputs):
        G = float(inputs.get("irradiance_W_m2", 600.0))
        demand = float(inputs.get("Q_demand_W", self.q_demand))
        f = self.solar_fraction(G)
        q_solar = f * demand
        q_aux_input = (demand - q_solar) / self.eta_boiler
        return {"f_solar": f, "Q_solar_W": q_solar,
                "Q_aux_input_W": q_aux_input, "irradiance_W_m2": G}
