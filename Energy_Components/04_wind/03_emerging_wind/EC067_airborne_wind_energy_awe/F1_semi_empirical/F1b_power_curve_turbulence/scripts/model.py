"""
EC067 — Airborne Wind Energy (AWE) System — F1b Power Curve + Turbulence Correction

Turbulence-corrected power output (second-order Taylor expansion):
    P_corr(V) = P(V) + 0.5 * d2P/dV2 * sigma_v^2,  sigma_v = TI * V

Air density at operational altitude using the international standard atmosphere
(ISA) barometric formula, with humidity correction via Buck (1981):
    T_h = T_0 - L * h          (temperature at altitude h)
    P_h = P_0 * (T_h/T_0)^(g/(R_d*L))    (ISA pressure)
    rho = (P_h / (Rd * T_h)) * (1 - 0.378 * e_w / P_h)

AWE-specific physics:
  - AWE operates at 200-600 m altitude where air density is 3-8% lower than
    ground level (e.g., 400 m: rho ~ 1.179 kg/m3 vs 1.225 at sea level)
  - Turbulence intensity is lower at altitude (0.05-0.12) than surface (0.10-0.25)
  - Power theoretically scales as P ~ rho * v^3 (Loyd model for pumping cycle)
  - The turbulence correction captures the nonlinear power-curve shape
    in the acceleration region (below rated wind speed)

References:
    Loyd (1980). "Crosswind kite power." J. Energy, 4(3), 106-111.
    Fagiano & Milanese (2012). "Airborne Wind Energy: Basic Concepts and
      Physical Foundations." Automatica 48(8), 1580-1588.
    Luchsinger (2013). "Pumping cycle kite power." Energies 6(2), 1502-1519.
    AWES Industry Report (2019). Airborne Wind Europe.
    Buck (1981). J. Appl. Meteorol. Climatol., 20(12), 1527-1532.
    IEC 61400-12-1:2017 — adapted to AWE context.
"""

import numpy as np
from scipy.interpolate import CubicSpline


class AirborneWindF1b:
    """AWE system — power curve with turbulence correction + altitude-corrected air density."""

    RHO_REF = 1.225  # Standard sea-level air density [kg/m3]
    G = 9.80665      # Gravitational acceleration [m/s2]

    def __init__(self, params: dict):
        s = params["system"]
        pc = params["power_curve"]
        atm = params["atmosphere"]

        self.rated_power = s["rated_power"]["value"]
        self.cut_in = s["cut_in_speed"]["value"]
        self.cut_out = s["cut_out_speed"]["value"]
        self.wing_area = s["wing_area"]["value"]
        self.op_altitude = s["operational_altitude_m"]["value"]
        self.glide_ratio = s["glide_ratio"]["value"]

        self.Rd = atm["Rd"]["value"]
        self.P_ground = atm["P_ground"]["value"]
        self.lapse_rate = atm["lapse_rate"]["value"]   # K/m
        self.T_ground_K = atm["T_ground_K"]["value"]   # K

        # Cubic spline for smooth d2P/dV2
        self._speeds = np.array(pc["wind_speeds_ms"], dtype=float)
        self._powers = np.array(pc["power_kw"], dtype=float)
        self._spline = CubicSpline(self._speeds, self._powers, bc_type="clamped")

    # ------------------------------------------------------------------
    # Atmosphere at altitude
    # ------------------------------------------------------------------

    @staticmethod
    def saturation_vapor_pressure(T_c):
        """Buck (1981) saturation vapor pressure (Pa)."""
        T_c = np.asarray(T_c, dtype=float)
        return 611.21 * np.exp(17.502 * T_c / (240.97 + T_c))

    def isa_temperature(self, altitude_m, T_ground_degC=None):
        """
        ISA temperature at altitude using standard lapse rate.
        If T_ground_degC given, offset the ISA profile.
        """
        h = np.asarray(altitude_m, dtype=float)
        if T_ground_degC is not None:
            T_ground_K = np.asarray(T_ground_degC, dtype=float) + 273.15
        else:
            T_ground_K = self.T_ground_K
        T_h = T_ground_K - self.lapse_rate * h
        return np.maximum(T_h, 216.65)  # Tropopause floor

    def isa_pressure(self, altitude_m, T_ground_degC=None):
        """
        ISA pressure at altitude using barometric formula:
        P_h = P_0 * (T_h / T_0)^(g / (Rd * L))
        """
        h = np.asarray(altitude_m, dtype=float)
        if T_ground_degC is not None:
            T0 = np.asarray(T_ground_degC, dtype=float) + 273.15
        else:
            T0 = self.T_ground_K
        T_h = self.isa_temperature(h, T_ground_degC)
        exponent = self.G / (self.Rd * self.lapse_rate)
        return self.P_ground * (T_h / T0) ** exponent

    def air_density(self, altitude_m=None, air_temperature_degC=None,
                    relative_humidity=0.0):
        """
        Moist air density at operational altitude.

        Uses ISA pressure at altitude, actual temperature, and humidity.

        Args:
            altitude_m: Altitude (m). Defaults to system operational altitude.
            air_temperature_degC: Ground-level temperature (degC). Used to offset ISA profile.
            relative_humidity: 0-1
        """
        if altitude_m is None:
            altitude_m = self.op_altitude
        h = np.asarray(altitude_m, dtype=float)

        # ISA-offset temperature at altitude
        T_h = self.isa_temperature(h, air_temperature_degC)
        P_h = self.isa_pressure(h, air_temperature_degC)

        RH = np.asarray(relative_humidity, dtype=float)
        T_c = T_h - 273.15
        e_w = RH * self.saturation_vapor_pressure(T_c)

        return (P_h / (self.Rd * T_h)) * (1.0 - 0.378 * e_w / P_h)

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
        """Second derivative d2P/dV2, zero outside operating range."""
        v = np.asarray(wind_speed, dtype=float)
        d2p = self._spline(v, 2)
        d2p = np.where(v < self.cut_in, 0.0, d2p)
        d2p = np.where(v > self.cut_out, 0.0, d2p)
        return d2p

    # ------------------------------------------------------------------
    # Main power method
    # ------------------------------------------------------------------

    def power(self, wind_speed, turbulence_intensity=0.0,
              altitude_m=None, air_temperature_degC=None,
              relative_humidity=0.0):
        """
        Turbulence-corrected AWE power with altitude air-density adjustment.

        P_corr = [P(V) + 0.5 * d2P/dV2 * (TI*V)^2] * (rho(h) / rho_ref)

        Args:
            wind_speed: Wind speed at operational altitude (m/s)
            turbulence_intensity: TI (0–0.20, AWE altitude sites typically 0.05-0.12)
            altitude_m: Operational altitude in metres (default: system param)
            air_temperature_degC: Ground-level temperature for ISA offset (degC)
            relative_humidity: 0-1

        Returns:
            Power in kW (clipped to [0, rated_power])
        """
        v = np.asarray(wind_speed, dtype=float)
        ti = np.asarray(turbulence_intensity, dtype=float)

        rho = self.air_density(altitude_m, air_temperature_degC, relative_humidity)

        p_base = self.power_curve(v)
        sigma_v = ti * v
        d2p = self.power_curve_d2(v)
        p_turb = p_base + 0.5 * d2p * sigma_v ** 2

        # Density scaling
        p_turb = p_turb * (rho / self.RHO_REF)

        p_turb = np.where(v < self.cut_in, 0.0, p_turb)
        p_turb = np.where(v > self.cut_out, 0.0, p_turb)
        return np.clip(p_turb, 0.0, self.rated_power)

    def capacity_factor(self, wind_speed, turbulence_intensity=0.0,
                        altitude_m=None, air_temperature_degC=None,
                        relative_humidity=0.0):
        """Capacity factor = P / P_rated."""
        return self.power(wind_speed, turbulence_intensity,
                          altitude_m, air_temperature_degC,
                          relative_humidity) / self.rated_power

    def loyd_limit_kw(self, wind_speed, altitude_m=None,
                      air_temperature_degC=None, relative_humidity=0.0):
        """
        Theoretical Loyd upper bound on AWE power (kW):
            P_Loyd = 2/27 * rho * A * v^3 * (L/D)^2

        Used to verify Cp-equivalent stays below physical limit.
        """
        v = np.asarray(wind_speed, dtype=float)
        rho = self.air_density(altitude_m, air_temperature_degC, relative_humidity)
        L_D = self.glide_ratio
        return (2.0 / 27.0) * rho * self.wing_area * v ** 3 * L_D ** 2 / 1000.0

    def turbulence_correction(self, wind_speed, turbulence_intensity,
                               altitude_m=None, air_temperature_degC=None,
                               relative_humidity=0.0):
        """Fractional CF correction from turbulence: (P_turb - P_base) / P_rated."""
        v = np.asarray(wind_speed, dtype=float)
        rho = self.air_density(altitude_m, air_temperature_degC, relative_humidity)
        p_base = self.power_curve(v) * (rho / self.RHO_REF)
        p_base = np.clip(p_base, 0.0, self.rated_power)
        p_turb = self.power(v, turbulence_intensity, altitude_m,
                            air_temperature_degC, relative_humidity)
        return (p_turb - p_base) / self.rated_power
