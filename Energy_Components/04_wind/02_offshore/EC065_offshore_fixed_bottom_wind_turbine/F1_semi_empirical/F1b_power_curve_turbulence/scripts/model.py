"""
EC065 — Offshore Fixed-Bottom Wind Turbine — F1b Turbulence + Humid Air Density

Same turbulence correction as EC062 F1b, plus air density correction for humidity:
    rho = P_atm / (Rd * T) * (1 - 0.378 * e_w / P_atm)

where e_w is the partial pressure of water vapour (Buck equation):
    e_w = RH * 611.21 * exp(17.502 * T_c / (240.97 + T_c))

Offshore turbines operate in marine environments with:
    - Lower TI (0.06-0.10 vs 0.10-0.25 onshore)
    - Higher humidity (reduces air density by ~1%)
    - Cooler, denser air overall

References:
    Albers et al. (2007). Wind Energy, 10(4), 395-406.
    Buck (1981). "New equations for computing vapor pressure and enhancement
    factor." J. Appl. Meteorol. Climatol., 20(12), 1527-1532.
    IEC 61400-12-1:2017.
"""

import numpy as np
from scipy.interpolate import CubicSpline


class OffshoreWindF1b:
    """Offshore wind turbine — turbulence correction + humid air density."""

    RHO_REF = 1.225  # Standard air density [kg/m3]

    def __init__(self, params: dict):
        t = params["turbine"]
        pc = params["power_curve"]
        atm = params["atmosphere"]

        self.rated_power = t["rated_power"]["value"]
        self.hub_height = t["hub_height"]["value"]
        self.rotor_diameter = t["rotor_diameter"]["value"]
        self.cut_in = t["cut_in_speed"]["value"]
        self.cut_out = t["cut_out_speed"]["value"]

        self.Rd = atm["Rd"]["value"]        # J/kgK
        self.P_atm = atm["P_atm"]["value"]  # Pa

        # Cubic spline for smooth d2P/dV2
        self._speeds = np.array(pc["wind_speeds_ms"], dtype=float)
        self._powers = np.array(pc["power_kw"], dtype=float)
        self._spline = CubicSpline(self._speeds, self._powers, bc_type='clamped')

        self.rotor_area = np.pi * (self.rotor_diameter / 2.0) ** 2

    @staticmethod
    def saturation_vapor_pressure(T_c):
        """Buck equation for saturation vapour pressure (Pa)."""
        T_c = np.asarray(T_c, dtype=float)
        return 611.21 * np.exp(17.502 * T_c / (240.97 + T_c))

    def humid_air_density(self, T_c, relative_humidity):
        """
        Air density corrected for humidity.
        rho = P / (Rd * T) * (1 - 0.378 * e_w / P)

        Args:
            T_c: Air temperature in degC
            relative_humidity: 0-1
        """
        T_c = np.asarray(T_c, dtype=float)
        RH = np.asarray(relative_humidity, dtype=float)
        T_K = T_c + 273.15

        e_sat = self.saturation_vapor_pressure(T_c)
        e_w = RH * e_sat

        rho = (self.P_atm / (self.Rd * T_K)) * (1.0 - 0.378 * e_w / self.P_atm)
        return rho

    def power_curve(self, wind_speed):
        """Base power from cubic spline (kW)."""
        v = np.asarray(wind_speed, dtype=float)
        p = self._spline(v)
        p = np.where(v < self.cut_in, 0.0, p)
        p = np.where(v > self.cut_out, 0.0, p)
        return np.clip(p, 0.0, self.rated_power)

    def power_curve_d2(self, wind_speed):
        """Second derivative d2P/dV2 from cubic spline."""
        v = np.asarray(wind_speed, dtype=float)
        d2p = self._spline(v, 2)
        d2p = np.where(v < self.cut_in, 0.0, d2p)
        d2p = np.where(v > self.cut_out, 0.0, d2p)
        return d2p

    def power(self, wind_speed, turbulence_intensity=0.0,
              air_temperature_degC=15.0, relative_humidity=0.5):
        """
        Turbulence-corrected power with humid air density.

        Args:
            wind_speed: m/s
            turbulence_intensity: TI (0-0.30)
            air_temperature_degC: degC
            relative_humidity: 0-1
        """
        v = np.asarray(wind_speed, dtype=float)
        ti = np.asarray(turbulence_intensity, dtype=float)

        # Air density from temperature and humidity
        rho = self.humid_air_density(air_temperature_degC, relative_humidity)

        # Base power + turbulence correction
        p_base = self.power_curve(v)
        sigma_v = ti * v
        d2p = self.power_curve_d2(v)
        p_turb = p_base + 0.5 * d2p * sigma_v**2

        # Density correction
        p_turb = p_turb * (rho / self.RHO_REF)

        # Enforce limits
        p_turb = np.where(v < self.cut_in, 0.0, p_turb)
        p_turb = np.where(v > self.cut_out, 0.0, p_turb)
        return np.clip(p_turb, 0.0, self.rated_power)

    def power_coefficient(self, wind_speed, turbulence_intensity=0.0,
                          air_temperature_degC=15.0, relative_humidity=0.5):
        """Cp = P / (0.5 * rho * A * v^3)."""
        v = np.asarray(wind_speed, dtype=float)
        rho = self.humid_air_density(air_temperature_degC, relative_humidity)
        p_kw = self.power(v, turbulence_intensity, air_temperature_degC, relative_humidity)
        p_avail = 0.5 * rho * self.rotor_area * v**3 / 1000.0
        return np.where(p_avail > 0, p_kw / p_avail, 0.0)
