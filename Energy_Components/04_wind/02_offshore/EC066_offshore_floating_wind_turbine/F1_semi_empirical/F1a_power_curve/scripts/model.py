"""
EC066 — Offshore Floating Wind Turbine — F1a Power Curve Model

Interpolated reference power curve P(v) with:
    - air density correction (IEC 61400-12-1)
    - platform motion penalty (1-3% efficiency loss from floater pitch/surge)

P(v, rho) = P_curve(v) * (rho / rho_ref) * (1 - eta_platform_loss)

Larger rotors (10-15 MW), higher hub heights, higher capacity factors (40-55%)
than fixed-bottom offshore due to better wind resource at deep-water sites.

Reference:
    Gaertner et al. (2020). "Definition of the IEA 15 MW Offshore Reference
    Wind Turbine," NREL/TP-5000-75698.
    Allen et al. (2020). "Definition of the UMaine VolturnUS-S Reference Platform,"
    NREL/TP-5000-76773.
"""

import numpy as np
from scipy.interpolate import interp1d


class FloatingOffshoreWindF1a:
    """Floating offshore wind turbine — power curve with platform motion penalty."""

    RHO_REF = 1.225  # kg/m3

    def __init__(self, params: dict):
        t = params["turbine"]
        pc = params["power_curve"]

        self.rated_power = t["rated_power"]["value"]  # kW
        self.hub_height = t["hub_height"]["value"]
        self.rotor_diameter = t["rotor_diameter"]["value"]
        self.cut_in = t["cut_in_speed"]["value"]
        self.cut_out = t["cut_out_speed"]["value"]
        self.platform_loss = t["platform_motion_penalty"]["value"]

        self._power_curve = interp1d(
            pc["wind_speeds_ms"], pc["power_kw"],
            kind="linear", bounds_error=False, fill_value=0.0
        )
        self.rotor_area = np.pi * (self.rotor_diameter / 2) ** 2

    def power(self, wind_speed, air_density=1.225):
        """
        Power output in kW including platform motion penalty.

        Args:
            wind_speed: Hub-height wind speed in m/s
            air_density: Air density in kg/m3 (default 1.225)
        """
        v = np.asarray(wind_speed, dtype=float)
        rho = np.asarray(air_density, dtype=float)

        p = self._power_curve(v)
        p = p * (rho / self.RHO_REF)
        p = p * (1.0 - self.platform_loss)
        p = np.where(v > self.cut_out, 0.0, p)
        return np.clip(p, 0.0, self.rated_power)

    def capacity_factor(self, wind_speed, air_density=1.225):
        """Instantaneous capacity factor = P / P_rated."""
        return self.power(wind_speed, air_density) / self.rated_power

    def power_coefficient(self, wind_speed, air_density=1.225):
        """Cp = P / (0.5 * rho * A * v^3). Betz limit = 0.593."""
        v = np.asarray(wind_speed, dtype=float)
        rho = np.asarray(air_density, dtype=float)
        p_kw = self.power(v, rho)
        p_avail = 0.5 * rho * self.rotor_area * v**3 / 1000.0  # kW
        return np.where(p_avail > 0, p_kw / p_avail, 0.0)

    @staticmethod
    def wind_shear(v_ref, h_ref, h_target, alpha=0.11):
        """Power law shear correction; offshore alpha ~0.11 (lower than onshore 0.143)."""
        return v_ref * (h_target / h_ref) ** alpha
