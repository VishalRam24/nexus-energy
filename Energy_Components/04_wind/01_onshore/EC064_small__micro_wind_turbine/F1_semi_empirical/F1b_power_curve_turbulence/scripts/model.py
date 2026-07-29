"""
EC064 — Small/Micro Wind Turbine — F1b Power Curve + Turbulence Correction

Turbulence-corrected power output (second-order Taylor expansion):
    P_corr(V) = P(V) + 0.5 * d2P/dV2 * sigma_v^2
    where sigma_v = TI * V

Air density corrected for temperature, pressure, and humidity:
    rho = (P / (Rd * T)) * (1 - 0.378 * e_w / P)
    e_w = RH * e_sat(T),  e_sat from Buck (1981) equation

Small/micro turbines differ from utility-scale turbines in that:
  - Higher site turbulence intensities (TI 0.15-0.40, often class C/D sites)
  - Smaller rotor inertia -> faster response to gusts but more scatter
  - Operating pressure can vary significantly for rooftop/altitude sites
  - Lower Reynolds number blades, slightly different d2P/dV2 shape

References:
    IEC 61400-2:2013 — Design requirements for small wind turbines.
    IEC 61400-12-1:2017 — Power performance measurement.
    Albers et al. (2007). "Turbulence and Shear Normalisation of Wind
      Turbine Power Curves." Wind Energy, 10(4), 395-406.
    Buck (1981). J. Appl. Meteorol. Climatol., 20(12), 1527-1532.
"""

import numpy as np
from scipy.interpolate import CubicSpline


class SmallWindTurbineF1b:
    """Small/micro wind turbine — power curve with turbulence + air-density corrections."""

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

        self.Rd = atm["Rd"]["value"]
        self.P_ref = atm["P_ref"]["value"]

        # Cubic spline for smooth d2P/dV2
        self._speeds = np.array(pc["wind_speeds_ms"], dtype=float)
        self._powers = np.array(pc["power_kw"], dtype=float)
        self._spline = CubicSpline(self._speeds, self._powers, bc_type="clamped")

        self.rotor_area = np.pi * (self.rotor_diameter / 2.0) ** 2

    # ------------------------------------------------------------------
    # Air density
    # ------------------------------------------------------------------

    @staticmethod
    def saturation_vapor_pressure(T_c):
        """Buck (1981) saturation vapor pressure (Pa)."""
        T_c = np.asarray(T_c, dtype=float)
        return 611.21 * np.exp(17.502 * T_c / (240.97 + T_c))

    def air_density(self, pressure_pa, air_temperature_degC, relative_humidity=0.0):
        """
        Moist-air density accounting for temperature, pressure, and humidity.

        rho = (P / (Rd * T)) * (1 - 0.378 * e_w / P)

        Args:
            pressure_pa: Site atmospheric pressure (Pa)
            air_temperature_degC: Air temperature (degC)
            relative_humidity: 0-1 (default 0 for dry air)
        """
        P = np.asarray(pressure_pa, dtype=float)
        T_c = np.asarray(air_temperature_degC, dtype=float)
        RH = np.asarray(relative_humidity, dtype=float)
        T_K = T_c + 273.15
        e_w = RH * self.saturation_vapor_pressure(T_c)
        return (P / (self.Rd * T_K)) * (1.0 - 0.378 * e_w / P)

    # ------------------------------------------------------------------
    # Power curve helpers
    # ------------------------------------------------------------------

    def power_curve(self, wind_speed):
        """Base power from cubic spline (kW), region-enforced."""
        v = np.asarray(wind_speed, dtype=float)
        p = self._spline(v)
        p = np.where(v < self.cut_in, 0.0, p)
        p = np.where(v > self.cut_out, 0.0, p)
        return np.clip(p, 0.0, self.rated_power)

    def power_curve_d2(self, wind_speed):
        """Second derivative d2P/dV2 from cubic spline, zero outside operating range."""
        v = np.asarray(wind_speed, dtype=float)
        d2p = self._spline(v, 2)
        d2p = np.where(v < self.cut_in, 0.0, d2p)
        d2p = np.where(v > self.cut_out, 0.0, d2p)
        return d2p

    # ------------------------------------------------------------------
    # Main power method
    # ------------------------------------------------------------------

    def power(self, wind_speed, turbulence_intensity=0.0,
              pressure_pa=101325.0, air_temperature_degC=15.0,
              relative_humidity=0.0):
        """
        Turbulence-corrected power with full air-density adjustment.

        P_corr = [P(V) + 0.5 * d2P/dV2 * (TI*V)^2] * (rho / rho_ref)

        Args:
            wind_speed: Hub-height wind speed (m/s)
            turbulence_intensity: TI = sigma_v / V (0–0.40, small wind sites up to 0.40)
            pressure_pa: Site atmospheric pressure (Pa). Default sea-level.
            air_temperature_degC: Air temperature (degC)
            relative_humidity: 0-1 (default 0)

        Returns:
            Power in kW (clipped to [0, rated_power])
        """
        v = np.asarray(wind_speed, dtype=float)
        ti = np.asarray(turbulence_intensity, dtype=float)

        rho = self.air_density(pressure_pa, air_temperature_degC, relative_humidity)

        # Base power from interpolated curve
        p_base = self.power_curve(v)

        # Turbulence correction (second-order Taylor)
        sigma_v = ti * v
        d2p = self.power_curve_d2(v)
        p_turb = p_base + 0.5 * d2p * sigma_v ** 2

        # Air-density scaling
        p_turb = p_turb * (rho / self.RHO_REF)

        # Enforce operating limits
        p_turb = np.where(v < self.cut_in, 0.0, p_turb)
        p_turb = np.where(v > self.cut_out, 0.0, p_turb)
        return np.clip(p_turb, 0.0, self.rated_power)

    def capacity_factor(self, wind_speed, turbulence_intensity=0.0,
                        pressure_pa=101325.0, air_temperature_degC=15.0,
                        relative_humidity=0.0):
        """Capacity factor = P / P_rated."""
        return self.power(wind_speed, turbulence_intensity,
                          pressure_pa, air_temperature_degC, relative_humidity) / self.rated_power

    def power_coefficient(self, wind_speed, turbulence_intensity=0.0,
                          pressure_pa=101325.0, air_temperature_degC=15.0,
                          relative_humidity=0.0):
        """Cp = P / (0.5 * rho * A * v^3)."""
        v = np.asarray(wind_speed, dtype=float)
        rho = self.air_density(pressure_pa, air_temperature_degC, relative_humidity)
        p_kw = self.power(v, turbulence_intensity, pressure_pa,
                          air_temperature_degC, relative_humidity)
        p_avail = 0.5 * rho * self.rotor_area * v ** 3 / 1000.0
        return np.where(p_avail > 0, p_kw / p_avail, 0.0)

    def turbulence_correction(self, wind_speed, turbulence_intensity,
                               pressure_pa=101325.0, air_temperature_degC=15.0,
                               relative_humidity=0.0):
        """
        Fractional power correction from turbulence relative to rated power.
        delta_CF = (P_turb - P_base) / P_rated
        """
        v = np.asarray(wind_speed, dtype=float)
        rho = self.air_density(pressure_pa, air_temperature_degC, relative_humidity)
        p_base = self.power_curve(v) * (rho / self.RHO_REF)
        p_base = np.clip(p_base, 0.0, self.rated_power)
        p_turb = self.power(v, turbulence_intensity, pressure_pa,
                            air_temperature_degC, relative_humidity)
        return (p_turb - p_base) / self.rated_power
