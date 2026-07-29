"""Empirical part-load efficiency curve for EC085 Natural Gas Boiler (Condensing).

eta(PLR) tabulated as breakpoints, interpolated with np.interp (1-D lookup).
Q_out = PLR * Q_rated;  fuel/electric input = Q_out / eta(PLR).
Source: EnergyPlus Engineering Reference (2023) Boiler:HotWater; Stafford (2009) Energy & Buildings 41(2)168-175
"""
import numpy as np


class EffCurveModel:
    def __init__(self, params):
        p = params["table"]
        self.plr = np.asarray(p["plr"], dtype=float)
        self.eta = np.asarray(p["eta"], dtype=float)
        self.q_rated = float(params["Q_rated"]["value"])
        self.plr_min = float(params["PLR_min"]["value"])
        self.eta_nom = float(params["eta_nom"]["value"])

    def efficiency(self, plr):
        plr = np.clip(plr, self.plr_min, 1.0)
        return float(np.interp(plr, self.plr, self.eta))

    def predict(self, inputs):
        plr = float(inputs.get("part_load_ratio", 1.0))
        eta = self.efficiency(plr)
        q_out = max(plr, 0.0) * self.q_rated
        q_in = q_out / eta if eta > 0 else 0.0
        return {"efficiency": eta, "Q_out_kW": q_out, "Q_in_kW": q_in,
                "part_load_ratio": float(np.clip(plr, 0.0, 1.0))}
