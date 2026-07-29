"""
EC070 — Water-Source Heat Pump — F1a COP Map Model

COP = eta_Carnot * COP_Carnot = eta_Carnot * T_sink / (T_sink - T_source)
Q_heating = COP * W_compressor
W_compressor = Q_rated * PLR / COP

Water source provides more stable temperatures (10-25 degC) than air,
yielding higher COP (4.5-6.0) compared to ASHP (2.5-4.0).

Reference:
    ASHRAE Handbook — HVAC Systems and Equipment (2020), Ch. 9.
    EN 14511 standard rating conditions.
"""

import numpy as np


class WaterSourceHPF1a:
    """Water-source heat pump — COP as a function of source/sink temperatures."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.rated_capacity = u["rated_capacity"]["value"]  # kW_th
        self.carnot_fraction = u["carnot_fraction"]["value"]
        self.aux_power = u["auxiliary_power"]["value"]  # kW

    def cop(self, T_source_c, T_sink_c):
        """COP (heating mode) from Carnot fraction approach."""
        T_source = np.asarray(T_source_c, dtype=float) + 273.15
        T_sink = np.asarray(T_sink_c, dtype=float) + 273.15
        dT = T_sink - T_source
        cop_carnot = np.where(dT > 0, T_sink / dT, 30.0)
        cop = self.carnot_fraction * cop_carnot
        return np.clip(cop, 1.0, 20.0)

    def cooling_cop(self, T_source_c, T_sink_c):
        """COP in cooling mode: COP_cool = COP_heat - 1."""
        cop_heat = self.cop(T_source_c, T_sink_c)
        return np.clip(cop_heat - 1.0, 0.5, 19.0)

    def heating_capacity(self, T_source_c, T_sink_c, plr=1.0):
        """Heating output in kW at given part-load ratio."""
        return self.rated_capacity * np.asarray(plr, dtype=float)

    def electrical_input(self, T_source_c, T_sink_c, plr=1.0):
        """Compressor + auxiliary electrical input in kW."""
        q = self.heating_capacity(T_source_c, T_sink_c, plr)
        c = self.cop(T_source_c, T_sink_c)
        return q / c + self.aux_power

    def cooling_capacity(self, T_source_c, T_sink_c, plr=1.0):
        """Cooling output Q_c = Q_h - W_elec."""
        q_h = self.heating_capacity(T_source_c, T_sink_c, plr)
        w = self.electrical_input(T_source_c, T_sink_c, plr)
        return q_h - w
