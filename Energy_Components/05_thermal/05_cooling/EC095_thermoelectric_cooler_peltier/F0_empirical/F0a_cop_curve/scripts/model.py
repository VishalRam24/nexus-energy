"""Empirical COP map for EC095 Thermoelectric Cooler (Peltier, Bi2Te3).

COP tabulated against a single driving variable (lift or driving temperature)
and read back with np.interp. The rated point reproduces the datasheet COP.
Source: Goldsmid (2010) Introduction to Thermoelectricity; Riffat & Ma (2003) ATE 23,913-935
"""
import numpy as np


class CopMapModel:
    def __init__(self, params):
        t = params["table"]
        self.x = np.asarray(t["x"], dtype=float)
        self.cop = np.asarray(t["cop"], dtype=float)
        self.x_key = t["x_key"]
        self.cop_rated = float(params["COP_rated"]["value"])
        self.q_rated = float(params["Q_rated"]["value"])

    def cop_at(self, x):
        return float(np.interp(x, self.x, self.cop))

    def predict(self, inputs):
        x = float(inputs.get(self.x_key, self.x[len(self.x) // 2]))
        plr = float(inputs.get("part_load_ratio", 1.0))
        cop = self.cop_at(x)
        q_cool = max(min(plr, 1.0), 0.0) * self.q_rated
        w_in = q_cool / cop if cop > 0 else 0.0
        return {"COP": cop, "Q_cool_kW": q_cool, "W_in_kW": w_in, self.x_key: x}
