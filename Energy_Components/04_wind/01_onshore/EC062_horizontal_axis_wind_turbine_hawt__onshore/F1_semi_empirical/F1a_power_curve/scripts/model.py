"""
EC062 — HAWT Onshore Wind Turbine — F1a Power Curve Model

Interpolated power curve P(v) with air density correction:
    P(v, rho) = P_curve(v) * (rho / rho_ref)

Reference:
    IEC 61400-12-1 standard power curve methodology.
    Manwell et al. (2009). "Wind Energy Explained," 2nd ed. Wiley.

Library:
    windpowerlib v0.2.2 (MIT license) for turbine database.
"""

import numpy as np
from scipy.interpolate import interp1d


class HAWTOnshoreF1a:
    """Onshore wind turbine — power curve with air density correction."""

    RHO_REF = 1.225  # Standard air density [kg/m3]

    def __init__(self, params: dict):
        t = params["turbine"]
        pc = params["power_curve"]

        self.rated_power = t["rated_power"]["value"]  # kW
        self.hub_height = t["hub_height"]["value"]
        self.rotor_diameter = t["rotor_diameter"]["value"]
        self.cut_in = t["cut_in_speed"]["value"]
        self.cut_out = t["cut_out_speed"]["value"]

        # Build interpolation function
        self._power_curve = interp1d(
            pc["wind_speeds_ms"], pc["power_kw"],
            kind="linear", bounds_error=False, fill_value=0.0
        )
        self.rotor_area = np.pi * (self.rotor_diameter / 2) ** 2

    def power(self, wind_speed, air_density=1.225):
        """
        Power output in kW.

        Args:
            wind_speed: Hub-height wind speed in m/s
            air_density: Air density in kg/m3 (default 1.225)
        """
        v = np.asarray(wind_speed, dtype=float)
        rho = np.asarray(air_density, dtype=float)

        # Power curve at reference density
        p = self._power_curve(v)

        # Density correction (IEC method)
        p = p * (rho / self.RHO_REF)

        # Enforce cut-out
        p = np.where(v > self.cut_out, 0.0, p)
        return np.clip(p, 0.0, self.rated_power)

    def capacity_factor(self, wind_speed, air_density=1.225):
        """Capacity factor = P / P_rated."""
        return self.power(wind_speed, air_density) / self.rated_power

    def power_coefficient(self, wind_speed, air_density=1.225):
        """Cp = P / (0.5 * rho * A * v^3). Betz limit = 0.593."""
        v = np.asarray(wind_speed, dtype=float)
        rho = np.asarray(air_density, dtype=float)
        p_kw = self.power(v, rho)
        p_avail = 0.5 * rho * self.rotor_area * v**3 / 1000.0  # kW
        return np.where(p_avail > 0, p_kw / p_avail, 0.0)

    @staticmethod
    def wind_shear(v_ref, h_ref, h_target, alpha=0.143):
        """Power law wind shear correction: v(h) = v_ref * (h/h_ref)^alpha."""
        return v_ref * (h_target / h_ref) ** alpha
