"""EC215 Solar Still / HDH F0a — empirical yield-vs-irradiance lookup.

Passive solar-still daily distillate productivity (L/m2/day) scales with the
incident solar irradiance. 1-D np.interp over a tabulated (irradiance, yield)
breakpoint array; total yield = productivity * collector area.

Source: Kaushal & Varun (2010); Narayan et al. (2012) (reused from EC215 F1a).
NumPy only.
"""
import numpy as np


class YieldLookup:
    def __init__(self, irr_bp, yield_bp, irr_rated, yield_rated,
                 collector_area_m2, gor_hdh, gor_min, gor_max, sec_solar_kWh_m3):
        self.irr_bp = np.asarray(irr_bp, dtype=float)
        self.yield_bp = np.asarray(yield_bp, dtype=float)
        self.irr_rated = float(irr_rated)
        self.yield_rated = float(yield_rated)
        self.collector_area_m2 = float(collector_area_m2)
        self.gor_hdh = float(gor_hdh)
        self.gor_min = float(gor_min)
        self.gor_max = float(gor_max)
        self.sec_solar_kWh_m3 = float(sec_solar_kWh_m3)

    def productivity(self, irradiance):
        """Productivity (L/m2/day) at given irradiance; clamps to endpoints."""
        return np.interp(irradiance, self.irr_bp, self.yield_bp)

    def daily_yield(self, irradiance, area=None):
        """Total daily distillate (L/day) = productivity * area."""
        if area is None:
            area = self.collector_area_m2
        return self.productivity(irradiance) * np.asarray(area, dtype=float)
