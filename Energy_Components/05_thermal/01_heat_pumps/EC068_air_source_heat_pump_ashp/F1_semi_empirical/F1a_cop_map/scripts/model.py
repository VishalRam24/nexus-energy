"""
EC068 — Air-Source Heat Pump — F1a COP Map Model

COP = eta_Carnot * COP_Carnot = eta_Carnot * T_sink / (T_sink - T_source)
Q_heating = COP * W_compressor
W_compressor = Q_rated * PLR / COP

Reference:
    Staffell et al. (2012). Energy Environ. Sci., 5, 9291-9306.
    EN 14511 standard rating conditions.
"""

import numpy as np


class ASHPF1a:
    """Air-source heat pump — COP as a function of source/sink temperatures."""

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
        cop_carnot = np.where(dT > 0, T_sink / dT, 20.0)
        cop = self.carnot_fraction * cop_carnot
        return np.clip(cop, 1.0, 15.0)

    def heating_capacity(self, T_source_c, T_sink_c, plr=1.0):
        """Heating output in kW at given part-load ratio."""
        return self.rated_capacity * np.asarray(plr, dtype=float)

    def electrical_input(self, T_source_c, T_sink_c, plr=1.0):
        """Compressor + auxiliary electrical input in kW."""
        q = self.heating_capacity(T_source_c, T_sink_c, plr)
        c = self.cop(T_source_c, T_sink_c)
        return q / c + self.aux_power

    def seasonal_cop(self, T_sources, T_sink_c, hours=None):
        """Seasonal COP (SCOP) from temperature time series."""
        T = np.asarray(T_sources, dtype=float)
        c = self.cop(T, T_sink_c)
        if hours is not None:
            hours = np.asarray(hours, dtype=float)
            q_total = np.sum(self.rated_capacity * hours)
            w_total = np.sum(self.rated_capacity / c * hours) + self.aux_power * np.sum(hours)
            return q_total / w_total
        return np.mean(c)
