"""
EC215 — Solar Still / Humidification-Dehumidification (HDH) — F1a SEC, Recovery

Two modes:
1. Solar Still: passive evaporation-condensation in sealed glazed enclosure
   productivity ~ 4-6 L/(m2*day), strongly dependent on irradiance
2. HDH: air humidified over heated seawater, dehumidified by cooling
   GOR = 1-3, driven by solar thermal energy

Model:
    Solar still: yield [L/h] = productivity * area * (G / G_ref)
    HDH: GOR(G) = GOR_ref * (G / G_ref)^0.5 (capped to [GOR_min, GOR_max])
         yield = GOR * Q_solar / h_vap  (h_vap ~ 2.45 MJ/kg)

References:
    Kaushal, A. & Varun (2010). Solar stills: A review. Renew. Sust. Energy Rev. 14:446-453.
    Narayan, G.P. et al. (2012). Thermodynamic analysis of humidification dehumidification
      desalination cycles. Desalin. Water Treat. 16:339-353.
"""

import numpy as np

H_VAP = 2.45e6  # J/kg (latent heat of vaporization at ~60 C)
HOURS_PER_DAY = 24.0


class SolarStillHDHF1a:
    """Solar still and HDH desalination model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.mode = u["mode"]["value"]
        self.GOR_ref = u["GOR_HDH"]["value"]
        self.GOR_min = u["GOR_min"]["value"]
        self.GOR_max = u["GOR_max"]["value"]
        self.productivity_ref = u["productivity_L_m2_day"]["value"]  # L/(m2*day)
        self.G_ref = u["solar_irradiance_W_m2"]["value"]
        self.area = u["collector_area_m2"]["value"]
        self.SEC_solar = u["SEC_solar_kWh_m3"]["value"]

    def GOR(self, solar_irradiance_W_m2=None):
        """HDH gain output ratio [-]."""
        G = self.G_ref if solar_irradiance_W_m2 is None else np.asarray(solar_irradiance_W_m2, dtype=float)
        gor = self.GOR_ref * (G / self.G_ref) ** 0.5
        return np.clip(gor, self.GOR_min, self.GOR_max)

    def solar_power(self, capacity_fraction, solar_irradiance_W_m2=None):
        """Solar thermal power absorbed [W]."""
        G = self.G_ref if solar_irradiance_W_m2 is None else np.asarray(solar_irradiance_W_m2, dtype=float)
        cf = np.asarray(capacity_fraction, dtype=float)
        return cf * self.area * G  # W

    def yield_L_h(self, capacity_fraction, solar_irradiance_W_m2=None):
        """
        Distillate yield [L/h].
        Solar still: productivity * area * irradiance scaling.
        HDH: from GOR * Q_solar / H_vap.
        """
        cf = np.asarray(capacity_fraction, dtype=float)
        G = self.G_ref if solar_irradiance_W_m2 is None else np.asarray(solar_irradiance_W_m2, dtype=float)

        if self.mode == "solar_still":
            # L/(m2*day) -> L/h * irradiance ratio
            y = cf * self.productivity_ref * self.area * (G / self.G_ref) / HOURS_PER_DAY
        else:  # HDH
            Q_W = cf * self.area * G  # W
            Q_Jh = Q_W * 3600.0       # J/h
            gor = self.GOR(G)
            yield_kg_h = gor * Q_Jh / H_VAP
            y = yield_kg_h  # kg/h ~ L/h (density ~ 1)
        return np.maximum(y, 0.0)

    def yield_m3_h(self, capacity_fraction, solar_irradiance_W_m2=None):
        """Distillate yield [m3/h]."""
        return self.yield_L_h(capacity_fraction, solar_irradiance_W_m2) / 1000.0

    def sec_solar_kWh_m3(self, solar_irradiance_W_m2=None):
        """Equivalent solar SEC [kWh/m3] — constant (design value)."""
        return float(self.SEC_solar)
