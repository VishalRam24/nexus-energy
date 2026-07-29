"""
EC212 — Multi-Stage Flash Distillation (MSF) — F1a SEC, Recovery

MSF: brine heated above boiling point then flashed into multiple stages at decreasing pressures.
GOR = kg distillate / kg steam input.

Model:
    distillate_flow [m3/h] = capacity_fraction * capacity_m3_h
    Q_thermal [kJ/h]       = distillate_flow [kg/h] * SEC_thermal_kJ_kg  (density ~1 kg/L)
    W_elec [kWh/h]         = distillate_flow * SEC_elec

    GOR correction for top brine temperature:
    GOR(T) = GOR_ref * (T / T_ref)^0.5  — simplified Dahl correlation

References:
    El-Dessouky, H.T. & Ettouney, H.M. (2002). Fundamentals of Salt Water Desalination.
      Elsevier.
    Ettouney, H. (2006). Design and analysis of humidification dehumidification
      desalination process. Desalination 183:341-352.
"""

import numpy as np

RHO_WATER = 1000.0  # kg/m3


class MSFF1a:
    """MSF thermal desalination SEC and recovery model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.GOR_ref = u["GOR"]["value"]
        self.SEC_thermal = u["SEC_thermal_kJ_kg"]["value"]  # kJ/kg
        self.SEC_elec = u["SEC_elec_kWh_m3"]["value"]       # kWh/m3
        self.T_top_ref = u["T_top_brine_C"]["value"]
        self.N_stages = u["N_stages"]["value"]
        self.recovery = u["recovery"]["value"]
        self.capacity_m3_h = u["capacity_m3_h"]["value"]

    def GOR(self, T_top_C=None):
        """Gain Output Ratio [-]."""
        T = self.T_top_ref if T_top_C is None else np.asarray(T_top_C, dtype=float)
        return self.GOR_ref * (T / self.T_top_ref) ** 0.5

    def distillate_flow(self, capacity_fraction):
        """Distillate (product water) flow [m3/h]."""
        cf = np.asarray(capacity_fraction, dtype=float)
        return cf * self.capacity_m3_h

    def thermal_energy(self, capacity_fraction, T_top_C=None):
        """Thermal energy input [GJ/h]."""
        dist_kg_h = self.distillate_flow(capacity_fraction) * RHO_WATER / 1000.0  # kg -> tonne equiv
        # kJ/kg * kg/h -> kJ/h -> GJ/h
        dist_kg_h_actual = self.distillate_flow(capacity_fraction) * RHO_WATER
        return dist_kg_h_actual * self.SEC_thermal / 1e6  # GJ/h

    def electric_power(self, capacity_fraction):
        """Electrical power [kWh/h]."""
        return self.distillate_flow(capacity_fraction) * self.SEC_elec

    def steam_consumption(self, capacity_fraction, T_top_C=None):
        """Steam (heat source) consumption [kg/h]."""
        dist_kg_h = self.distillate_flow(capacity_fraction) * RHO_WATER
        gor = self.GOR(T_top_C)
        return dist_kg_h / gor
