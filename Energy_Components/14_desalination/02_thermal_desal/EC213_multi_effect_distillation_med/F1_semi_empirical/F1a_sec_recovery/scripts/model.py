"""
EC213 — Multi-Effect Distillation (MED) — F1a SEC, Recovery

MED: seawater evaporated in series of effects at decreasing temperatures/pressures.
Lower T_top than MSF: less scaling, lower corrosion, but requires vacuum system.

Model:
    GOR(N) = GOR_ref * (N / N_ref)^0.8  — power law with number of effects
    distillate_flow = capacity_fraction * capacity_m3_h
    Q_thermal = distillate_flow [kg/h] * SEC_thermal
    W_elec = distillate_flow * SEC_elec

References:
    El-Dessouky, H.T. & Ettouney, H.M. (2002). Fundamentals of Salt Water Desalination.
    Al-Sahali, M. & Ettouney, H. (2007). Developments in thermal desalination processes.
      Desalination 214:227-240.
"""

import numpy as np

RHO_WATER = 1000.0


class MEDF1a:
    """MED thermal desalination model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.GOR_ref = u["GOR"]["value"]
        self.N_ref = u["N_effects"]["value"]
        self.SEC_thermal = u["SEC_thermal_kJ_kg"]["value"]
        self.SEC_elec = u["SEC_elec_kWh_m3"]["value"]
        self.T_top_ref = u["T_top_C"]["value"]
        self.recovery = u["recovery"]["value"]
        self.capacity_m3_h = u["capacity_m3_h"]["value"]

    def GOR(self, N_effects=None):
        """Gain Output Ratio [-]."""
        N = self.N_ref if N_effects is None else np.asarray(N_effects, dtype=float)
        return self.GOR_ref * (N / self.N_ref) ** 0.8

    def distillate_flow(self, capacity_fraction):
        """Distillate flow [m3/h]."""
        cf = np.asarray(capacity_fraction, dtype=float)
        return cf * self.capacity_m3_h

    def thermal_energy(self, capacity_fraction):
        """Thermal energy [GJ/h]."""
        dist_kg_h = self.distillate_flow(capacity_fraction) * RHO_WATER
        return dist_kg_h * self.SEC_thermal / 1e6  # GJ/h

    def electric_power(self, capacity_fraction):
        """Electrical power [kWh/h]."""
        return self.distillate_flow(capacity_fraction) * self.SEC_elec

    def steam_consumption(self, capacity_fraction, N_effects=None):
        """Steam consumption [kg/h]."""
        dist_kg_h = self.distillate_flow(capacity_fraction) * RHO_WATER
        gor = self.GOR(N_effects)
        return dist_kg_h / gor
