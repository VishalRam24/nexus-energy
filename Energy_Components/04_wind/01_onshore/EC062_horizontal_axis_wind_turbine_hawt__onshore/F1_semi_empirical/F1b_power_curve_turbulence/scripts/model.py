"""
EC062 — HAWT Onshore Wind Turbine — F1b Power Curve + Turbulence Correction

Turbulence-corrected power output:
    P_corr(V) = P(V) + 0.5 * d2P/dV2 * sigma_v^2
    where sigma_v = TI * V (turbulence standard deviation)

The second derivative of the power curve captures how turbulence-induced
velocity fluctuations affect mean power. In the concave-up region
(below rated), turbulence increases power; in the concave-down region
(near rated), it decreases power.

Uses cubic spline interpolation for smooth second derivative calculation.

References:
    Albers et al. (2007). "Turbulence and Shear Normalisation of Wind
    Turbine Power Curves." Wind Energy, 10(4), 395-406.
    IEC 61400-12-1:2017.
"""

import numpy as np
from scipy.interpolate import CubicSpline


class HAWTOnshoreF1b:
    """Onshore wind turbine — power curve with turbulence correction."""

    RHO_REF = 1.225  # Standard air density [kg/m3]

    def __init__(self, params: dict):
        t = params["turbine"]
        pc = params["power_curve"]

        self.rated_power = t["rated_power"]["value"]
        self.hub_height = t["hub_height"]["value"]
        self.rotor_diameter = t["rotor_diameter"]["value"]
        self.cut_in = t["cut_in_speed"]["value"]
        self.cut_out = t["cut_out_speed"]["value"]

        # Build cubic spline for power curve (gives smooth d2P/dV2)
        self._speeds = np.array(pc["wind_speeds_ms"], dtype=float)
        self._powers = np.array(pc["power_kw"], dtype=float)
        self._spline = CubicSpline(self._speeds, self._powers, bc_type='clamped')

        self.rotor_area = np.pi * (self.rotor_diameter / 2.0) ** 2

    def power_curve(self, wind_speed):
        """Base power from cubic spline interpolation (kW)."""
        v = np.asarray(wind_speed, dtype=float)
        p = self._spline(v)
        p = np.where(v < self.cut_in, 0.0, p)
        p = np.where(v > self.cut_out, 0.0, p)
        return np.clip(p, 0.0, self.rated_power)

    def power_curve_d2(self, wind_speed):
        """Second derivative of power curve d2P/dV2 from cubic spline."""
        v = np.asarray(wind_speed, dtype=float)
        d2p = self._spline(v, 2)  # Second derivative
        # Zero outside operating range
        d2p = np.where(v < self.cut_in, 0.0, d2p)
        d2p = np.where(v > self.cut_out, 0.0, d2p)
        return d2p

    def power(self, wind_speed, turbulence_intensity=0.0, air_density=1.225):
        """
        Turbulence-corrected power output in kW.

        P_corr = P(V) + 0.5 * d2P/dV2 * sigma_v^2
        sigma_v = TI * V

        Args:
            wind_speed: Hub-height wind speed (m/s)
            turbulence_intensity: TI = sigma_v / V (0-0.30)
            air_density: kg/m3 (default 1.225)
        """
        v = np.asarray(wind_speed, dtype=float)
        ti = np.asarray(turbulence_intensity, dtype=float)
        rho = np.asarray(air_density, dtype=float)

        # Base power from curve
        p_base = self.power_curve(v)

        # Turbulence correction
        sigma_v = ti * v
        d2p = self.power_curve_d2(v)
        p_turb = p_base + 0.5 * d2p * sigma_v**2

        # Air density correction
        p_turb = p_turb * (rho / self.RHO_REF)

        # Enforce physical limits
        p_turb = np.where(v < self.cut_in, 0.0, p_turb)
        p_turb = np.where(v > self.cut_out, 0.0, p_turb)
        return np.clip(p_turb, 0.0, self.rated_power)

    def capacity_factor(self, wind_speed, turbulence_intensity=0.0, air_density=1.225):
        """Capacity factor = P / P_rated."""
        return self.power(wind_speed, turbulence_intensity, air_density) / self.rated_power

    def power_coefficient(self, wind_speed, turbulence_intensity=0.0, air_density=1.225):
        """Cp = P / (0.5 * rho * A * v^3)."""
        v = np.asarray(wind_speed, dtype=float)
        rho = np.asarray(air_density, dtype=float)
        p_kw = self.power(v, turbulence_intensity, rho)
        p_avail = 0.5 * rho * self.rotor_area * v**3 / 1000.0
        return np.where(p_avail > 0, p_kw / p_avail, 0.0)

    def capacity_factor_correction(self, wind_speed, turbulence_intensity, air_density=1.225):
        """
        Fractional correction to capacity factor from turbulence.
        CF_corr = (P_turb - P_base) / P_rated
        """
        v = np.asarray(wind_speed, dtype=float)
        p_base = self.power_curve(v) * (np.asarray(air_density) / self.RHO_REF)
        p_base = np.clip(p_base, 0.0, self.rated_power)
        p_turb = self.power(v, turbulence_intensity, air_density)
        return (p_turb - p_base) / self.rated_power
