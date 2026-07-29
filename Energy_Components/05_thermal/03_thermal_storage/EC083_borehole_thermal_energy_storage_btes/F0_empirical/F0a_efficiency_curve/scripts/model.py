"""Empirical storage recovery-efficiency lookup for EC083 Borehole Thermal Energy Storage (BTES).

eta_store tabulated against storage temperature and read with np.interp.
Energy returned = eta_store * energy charged.
Source: Nordell (1994) Borehole Heat Store Design Optimization, Lulea Univ.
"""
import numpy as np


class StoreEffModel:
    def __init__(self, params):
        t = params["table"]
        self.x = np.asarray(t["x"], dtype=float)
        self.eta = np.asarray(t["eta"], dtype=float)
        self.x_key = t["x_key"]
        self.eta_rated = float(params["eta_rated"]["value"])
        self.cap_kwh = float(params["capacity_kWh"]["value"])

    def efficiency(self, x):
        return float(np.interp(x, self.x, self.eta))

    def predict(self, inputs):
        x = float(inputs.get(self.x_key, self.x[len(self.x) // 2]))
        e_charge = float(inputs.get("E_charge_kWh", self.cap_kwh))
        eta = self.efficiency(x)
        return {"efficiency": eta, "E_charge_kWh": e_charge,
                "E_discharge_kWh": eta * e_charge, self.x_key: x}
