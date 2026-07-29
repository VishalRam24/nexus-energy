"""
EC069 — Ground-Source Heat Pump (GSHP) — F1a COP Map Model

COP = eta_Carnot * COP_Carnot = eta_Carnot * T_sink / (T_sink - T_source)

Key difference vs ASHP (EC068):
  - Higher Carnot fraction (0.50 vs 0.45) due to stable ground source
  - Narrower, more stable T_source range (0–20°C vs -20–40°C for air)
  - No defrost cycle (ground loop antifreeze keeps fluid above 0°C)
  - Rated COP typically 4–6 vs 2.5–4 for ASHP at same sink temp

Rating condition: G10/W35 (ground 10°C in, water out 35°C)

References:
    Staffell et al. (2012). Energy Environ. Sci., 5, 9291-9306.
    ASHRAE Handbook — HVAC Applications (2019), Chapter 34.
    EN 15450 / ISO 13256-2 rating standards for GSHPs.
"""

import numpy as np


class GSHPF1a:
    """Ground-source heat pump — COP as a function of source/sink temperatures."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.rated_capacity = u["rated_capacity"]["value"]     # kW_th
        self.carnot_fraction = u["carnot_fraction"]["value"]   # eta_Carnot
        self.aux_power = u["auxiliary_power"]["value"]         # kW

    def cop(self, T_source_c, T_sink_c):
        """
        COP (heating mode) from Carnot fraction approach.

        COP = eta_Carnot * T_sink_K / (T_sink_K - T_source_K)

        Higher than ASHP because:
        1) Higher eta_Carnot (0.50 vs 0.45) — stable source reduces compressor cycling losses
        2) Smaller temperature lift at same sink (ground ~10°C, air ~0–7°C in heating season)
        """
        T_source = np.asarray(T_source_c, dtype=float) + 273.15
        T_sink   = np.asarray(T_sink_c,   dtype=float) + 273.15
        dT = T_sink - T_source
        cop_carnot = np.where(dT > 0, T_sink / dT, 20.0)
        cop = self.carnot_fraction * cop_carnot
        return np.clip(cop, 1.0, 20.0)  # GSHP can reach COP>10 in theory

    def heating_capacity(self, T_source_c, T_sink_c, plr=1.0):
        """Heating thermal output in kW at given part-load ratio."""
        return self.rated_capacity * np.asarray(plr, dtype=float)

    def electrical_input(self, T_source_c, T_sink_c, plr=1.0):
        """Compressor + pump auxiliary electrical input in kW."""
        q = self.heating_capacity(T_source_c, T_sink_c, plr)
        c = self.cop(T_source_c, T_sink_c)
        return q / c + self.aux_power

    def cop_advantage_over_ashp(self, T_source_c, T_sink_c, ashp_carnot_fraction=0.45):
        """
        Ratio of GSHP COP to hypothetical ASHP COP at same conditions.
        Always > 1 because higher eta_Carnot.
        """
        cop_gshp = self.cop(T_source_c, T_sink_c)
        T_source = np.asarray(T_source_c, dtype=float) + 273.15
        T_sink   = np.asarray(T_sink_c,   dtype=float) + 273.15
        dT = T_sink - T_source
        cop_ashp = ashp_carnot_fraction * np.where(dT > 0, T_sink / dT, 20.0)
        cop_ashp = np.clip(cop_ashp, 1.0, 15.0)
        return cop_gshp / cop_ashp
