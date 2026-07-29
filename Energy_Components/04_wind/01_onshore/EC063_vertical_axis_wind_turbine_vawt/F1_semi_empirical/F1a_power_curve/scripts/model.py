"""
EC063 — Vertical Axis Wind Turbine (VAWT) — F1a Power Curve Model

P(v, rho) = 0.5 * rho * A_swept * Cp(v) * v^3, with VAWT-typical Cp (0.20-0.35).
Omnidirectional — no yaw correction; insensitive to wind direction.
Lower Cp than HAWT but works in turbulent/urban environments.

Reference:
    Tjiu et al. (2015). Renew. Energy 75, 50-67.
    Sutherland et al. (2012). "The Sandia 100 m All-Glass Baseline Wind Turbine."
"""

import numpy as np
from scipy.interpolate import interp1d


class VAWTF1a:
    """Vertical-axis wind turbine — semi-empirical power curve."""

    RHO_REF = 1.225  # Standard air density [kg/m3]

    def __init__(self, params: dict):
        t = params["turbine"]
        pc = params["power_curve"]

        self.rated_power = t["rated_power"]["value"]  # kW
        self.hub_height = t["hub_height"]["value"]
        self.rotor_diameter = t["rotor_diameter"]["value"]
        self.rotor_height = t["rotor_height"]["value"]
        self.swept_area = t["swept_area"]["value"]
        self.cut_in = t["cut_in_speed"]["value"]
        self.cut_out = t["cut_out_speed"]["value"]
        self.max_cp = t["max_cp"]["value"]

        self._power_curve = interp1d(
            pc["wind_speeds_ms"], pc["power_kw"],
            kind="linear", bounds_error=False, fill_value=0.0
        )

    def power(self, wind_speed, air_density=1.225):
        """
        Power output in kW. Omnidirectional — no wind direction needed.

        Args:
            wind_speed: Wind speed in m/s
            air_density: Air density in kg/m3 (default 1.225)
        """
        v = np.asarray(wind_speed, dtype=float)
        rho = np.asarray(air_density, dtype=float)

        p = self._power_curve(v)
        p = p * (rho / self.RHO_REF)
        p = np.where(v < self.cut_in, 0.0, p)
        p = np.where(v > self.cut_out, 0.0, p)
        return np.clip(p, 0.0, self.rated_power)

    def capacity_factor(self, wind_speed, air_density=1.225):
        """Capacity factor = P / P_rated."""
        return self.power(wind_speed, air_density) / self.rated_power

    def power_coefficient(self, wind_speed, air_density=1.225):
        """Cp = P / (0.5 * rho * A * v^3). VAWT typical max ~0.32; Betz = 0.593."""
        v = np.asarray(wind_speed, dtype=float)
        rho = np.asarray(air_density, dtype=float)
        p_kw = self.power(v, rho)
        p_avail = 0.5 * rho * self.swept_area * v**3 / 1000.0  # kW
        return np.where(p_avail > 0, p_kw / p_avail, 0.0)
