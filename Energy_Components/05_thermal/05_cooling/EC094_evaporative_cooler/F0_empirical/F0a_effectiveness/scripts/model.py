"""Empirical wet-bulb effectiveness model for EC094 Evaporative Cooler (Direct).

T_out = T_db - eps*(T_db - T_wb), with eps a constant rated effectiveness.
Q_cool = m_air * Cp * (T_db - T_out). EER-like COP = Q_cool / P_fan.
Source: EC094 F1a effectiveness params; ASHRAE Handbook Fundamentals (2021) Ch.41 Evaporative Cooling
"""
import numpy as np  # noqa: F401


class EvapModel:
    def __init__(self, params):
        self.eps = float(params["epsilon"]["value"])
        self.cp = float(params["Cp_air_J_kgK"]["value"])
        self.rho = float(params["rho_air_kg_m3"]["value"])
        self.p_fan = float(params["P_fan_W"]["value"])

    def predict(self, inputs):
        t_db = float(inputs.get("T_db_C", 35.0))
        t_wb = float(inputs.get("T_wb_C", 20.0))
        m_air = float(inputs.get("m_air_kg_s", 1.0))
        t_out = t_db - self.eps * (t_db - t_wb)
        q_cool = m_air * self.cp * (t_db - t_out)
        cop = q_cool / self.p_fan if self.p_fan > 0 else 0.0
        return {"T_out_C": t_out, "Q_cool_W": q_cool, "effectiveness": self.eps,
                "COP": cop}
