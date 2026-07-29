"""
EC141 -- Anaerobic Digester (Thermophilic) -- F1a Yield Model

Simplest semi-empirical model:
    biogas_m3_per_day = feedstock_VS_kg_per_day * VS_destruction * biogas_yield
    CH4_m3_per_day    = biogas_m3_per_day * CH4_fraction
    energy_kWh_per_day = CH4_m3_per_day * LHV_methane_kWh_m3

Thermophilic (55 degC), HRT=15 days, VS destruction=65%.
F1b adds feedstock-specific BMPs, C/N ratio corrections, and temperature inhibition.

References:
    Angelidaki, I. et al. (2009). Water Sci. Technol. 59(5):927.
    Mata-Alvarez, J. et al. (2014). Renew. Sustain. Energy Rev. 36:412.
"""

import numpy as np


class AnaerobicDigesterThermophilicF1a:
    """Thermophilic AD: biogas = feedstock_VS * VS_destruction * biogas_yield."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.biogas_yield    = float(u["biogas_yield_m3_per_kgVS"]["value"])   # m3/kgVS
        self.CH4_fraction    = float(u["CH4_fraction"]["value"])
        self.HRT_days        = float(u["HRT_days"]["value"])
        self.VS_destruction  = float(u["VS_destruction"]["value"])
        self.T_operating     = float(u["T_operating_degC"]["value"])
        self.LHV_CH4_kWh_m3  = float(u["LHV_methane_kWh_m3"]["value"])

    def predict(self, feedstock_VS_kg_per_day: float) -> dict:
        """
        Parameters
        ----------
        feedstock_VS_kg_per_day : float
            Volatile solids input rate [kgVS/day]

        Returns
        -------
        dict with:
            VS_destroyed_kg_per_day   : VS destroyed [kgVS/day]
            biogas_m3_per_day         : total biogas production [m3/day]
            CH4_m3_per_day            : methane production [m3/day]
            CO2_m3_per_day            : CO2 production [m3/day]
            energy_kWh_per_day        : energy content of CH4 [kWh/day]
            CH4_fraction              : methane content [-]
        """
        VS_in  = float(np.clip(feedstock_VS_kg_per_day, 0.0, None))
        VS_des = VS_in * self.VS_destruction
        biogas = VS_des * self.biogas_yield
        CH4    = biogas * self.CH4_fraction
        CO2    = biogas * (1.0 - self.CH4_fraction)
        energy = CH4 * self.LHV_CH4_kWh_m3

        return {
            "VS_destroyed_kg_per_day": float(VS_des),
            "biogas_m3_per_day":       float(biogas),
            "CH4_m3_per_day":          float(CH4),
            "CO2_m3_per_day":          float(CO2),
            "energy_kWh_per_day":      float(energy),
            "CH4_fraction":            float(self.CH4_fraction),
        }
