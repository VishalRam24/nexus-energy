"""
EC065 — Offshore Fixed-Bottom Wind Turbine — F1a Power Curve Model

Interpolated power curve P(v) with air density correction:
    P(v, rho) = P_curve(v) * (rho / rho_ref)

Cut-in / cut-out enforced per IEC 61400-12-1.
Turbine: Siemens SWT-3.6-120 (3.6 MW, 120 m rotor, 90 m hub height).

Reference:
    IEC 61400-12-1:2017 — Wind turbines, Part 12-1: Power performance.
    windpowerlib (MIT): https://github.com/wind-python/windpowerlib
    Siemens SWT-3.6-120 product datasheet.
"""

import numpy as np
from scipy.interpolate import interp1d


class OffshoreWindF1a:
    """Offshore fixed-bottom wind turbine — power curve with air density correction."""

    RHO_REF = 1.225  # Standard air density [kg/m3]

    def __init__(self, params: dict):
        t = params["turbine"]
        pc = params["power_curve"]

        self.rated_power = t["rated_power"]["value"]    # kW
        self.hub_height = t["hub_height"]["value"]      # m
        self.rotor_diameter = t["rotor_diameter"]["value"]  # m
        self.cut_in = t["cut_in_speed"]["value"]        # m/s
        self.cut_out = t["cut_out_speed"]["value"]      # m/s

        # Interpolation function from manufacturer power curve
        self._power_curve = interp1d(
            pc["wind_speeds_ms"], pc["power_kw"],
            kind="linear", bounds_error=False, fill_value=0.0,
        )
        self.rotor_area = np.pi * (self.rotor_diameter / 2.0) ** 2  # m2

    def power(self, wind_speed, air_density=1.225):
        """
        Power output in kW.

        Args:
            wind_speed:  Hub-height wind speed in m/s.
            air_density: Air density in kg/m3 (default 1.225 for offshore).

        Returns:
            Power in kW (0 outside cut-in/cut-out, clipped to rated_power).
        """
        v = np.asarray(wind_speed, dtype=float)
        rho = np.asarray(air_density, dtype=float)

        # Read power from curve at reference density
        p = self._power_curve(v)

        # Linear density correction (IEC 61400-12-1, Annex L)
        p = p * (rho / self.RHO_REF)

        # Enforce cut-out (turbine stops above cut-out)
        p = np.where(v > self.cut_out, 0.0, p)

        return np.clip(p, 0.0, self.rated_power)

    def capacity_factor(self, wind_speed, air_density=1.225):
        """Capacity factor = P / P_rated."""
        return self.power(wind_speed, air_density) / self.rated_power

    def power_coefficient(self, wind_speed, air_density=1.225):
        """
        Power coefficient Cp = P / P_wind = P / (0.5 * rho * A * v^3).
        Betz limit = 0.593.
        """
        v = np.asarray(wind_speed, dtype=float)
        rho = np.asarray(air_density, dtype=float)
        p_kw = self.power(v, rho)
        p_avail_kw = 0.5 * rho * self.rotor_area * v ** 3 / 1000.0
        safe_avail = np.where(p_avail_kw > 0, p_avail_kw, 1.0)
        return np.where(p_avail_kw > 0, p_kw / safe_avail, 0.0)

    @staticmethod
    def wind_shear(v_ref, h_ref, h_target, alpha=0.11):
        """
        Power law wind shear correction: v(h) = v_ref * (h/h_ref)^alpha.

        Offshore alpha typically 0.10–0.14 (lower than onshore ~0.143).
        Default alpha=0.11 for open-water sites.
        """
        return v_ref * (h_target / h_ref) ** alpha
