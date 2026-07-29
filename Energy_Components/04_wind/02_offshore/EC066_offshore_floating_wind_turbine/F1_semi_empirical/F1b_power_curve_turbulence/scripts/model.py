"""
EC066 — Offshore Floating Wind Turbine — F1b Turbulence + Platform Pitch

Extends F1a (power curve + static platform penalty) with:

1. Marine humid air density (Buck equation, same as EC065 F1b):
       rho = P / (Rd * T) * (1 - 0.378 * e_w / P)

2. Wave-induced platform pitch angle penalty on rotor alignment:
       P_pitch = P_base * cos^2(theta)
   Rotor swept area exposed to the wind component is A * cos(theta),
   so available kinetic energy scales as (v * cos(theta))^3 / cos(theta)
   => effective P ~ P_base * cos^2(theta).
   This is conservative vs. full 3D aeroelastic but well-suited for F1b.
   For small angles (<5 deg) the correction is ~1-3% consistent with
   published FOWT field measurement data.

3. Offshore turbulence correction (second-order Taylor expansion):
       P_turb = P_base + 0.5 * d2P/dv2 * sigma_v^2
   with offshore TI default 0.07 (lower than onshore).

4. All three effects combined, with rated-power cap.

References:
    Gaertner et al. (2020). NREL/TP-5000-75698 (IEA 15 MW).
    Allen et al. (2020). NREL/TP-5000-76773 (UMaine VolturnUS-S).
    Buck (1981). J. Appl. Meteorol. Climatol., 20(12), 1527-1532.
    Jonkman et al. (2020). "OpenFAST" floating rotor aerodynamics.
    Jiang et al. (2014). Renew. Energy 72, 237-248. (pitch-induced power loss)
"""

import numpy as np
from scipy.interpolate import CubicSpline


class FloatingOffshoreWindF1b:
    """Offshore floating wind turbine — turbulence + platform pitch + humid air."""

    RHO_REF = 1.225  # kg/m3

    def __init__(self, params: dict):
        t = params["turbine"]
        pc = params["power_curve"]
        atm = params["atmosphere"]

        self.rated_power = t["rated_power"]["value"]
        self.hub_height = t["hub_height"]["value"]
        self.rotor_diameter = t["rotor_diameter"]["value"]
        self.cut_in = t["cut_in_speed"]["value"]
        self.cut_out = t["cut_out_speed"]["value"]
        self.static_platform_loss = t["platform_motion_penalty"]["value"]

        self.Rd = atm["Rd"]["value"]
        self.P_atm = atm["P_atm"]["value"]

        speeds = np.array(pc["wind_speeds_ms"], dtype=float)
        powers = np.array(pc["power_kw"], dtype=float)
        # CubicSpline for d2P/dv2 (turbulence correction)
        self._spline = CubicSpline(speeds, powers, bc_type="clamped")
        self.rotor_area = np.pi * (self.rotor_diameter / 2.0) ** 2

    # ------------------------------------------------------------------
    # Atmosphere
    # ------------------------------------------------------------------

    @staticmethod
    def saturation_vapor_pressure(T_c):
        """Buck equation for e_sat (Pa)."""
        T_c = np.asarray(T_c, dtype=float)
        return 611.21 * np.exp(17.502 * T_c / (240.97 + T_c))

    def humid_air_density(self, T_c, relative_humidity):
        """
        Marine air density: rho = P/(Rd*T) * (1 - 0.378*e_w/P)
        Offshore humidity 70-100% reduces density by 0.5-1.5%.
        """
        T_c = np.asarray(T_c, dtype=float)
        RH = np.asarray(relative_humidity, dtype=float)
        T_K = T_c + 273.15
        e_w = RH * self.saturation_vapor_pressure(T_c)
        return (self.P_atm / (self.Rd * T_K)) * (1.0 - 0.378 * e_w / self.P_atm)

    # ------------------------------------------------------------------
    # Power curve
    # ------------------------------------------------------------------

    def _power_spline(self, v):
        """P from cubic spline, region-enforced."""
        v = np.asarray(v, dtype=float)
        p = self._spline(v)
        p = np.where(v < self.cut_in, 0.0, p)
        p = np.where(v > self.cut_out, 0.0, p)
        return np.clip(p, 0.0, self.rated_power)

    def _d2p_spline(self, v):
        """Second derivative d2P/dv2, zero outside operating range."""
        v = np.asarray(v, dtype=float)
        d2p = self._spline(v, 2)
        d2p = np.where(v < self.cut_in, 0.0, d2p)
        d2p = np.where(v > self.cut_out, 0.0, d2p)
        return d2p

    # ------------------------------------------------------------------
    # Pitch penalty
    # ------------------------------------------------------------------

    @staticmethod
    def platform_pitch_factor(pitch_deg):
        """
        cos^2(theta) factor for wave-induced platform pitch.

        Physical basis: rotor normal tilted by theta from wind direction.
        Wind component perpendicular to rotor disc is v*cos(theta), so
        kinetic energy flux through rotor scales as cos^2(theta) times
        the projected area correction -> net factor = cos^2(theta).

        At theta=0: factor=1.0 (no penalty).
        At theta=5 deg: factor=0.9924 (~0.8% loss).
        At theta=10 deg: factor=0.9698 (~3% loss).
        """
        theta_rad = np.asarray(pitch_deg, dtype=float) * np.pi / 180.0
        return np.cos(theta_rad) ** 2

    # ------------------------------------------------------------------
    # Combined power
    # ------------------------------------------------------------------

    def power(self, wind_speed, turbulence_intensity=0.07,
              platform_pitch_deg=0.0,
              air_temperature_degC=12.0, relative_humidity=0.80):
        """
        Full F1b power with:
          (a) humid air density,
          (b) turbulence correction (Taylor d2P/dv2),
          (c) platform pitch penalty (cos^2 theta).

        Args:
            wind_speed: m/s
            turbulence_intensity: TI 0–0.20 (offshore default 0.07)
            platform_pitch_deg: platform pitch angle in degrees (0 = upright)
            air_temperature_degC: degC
            relative_humidity: 0–1
        """
        v = np.asarray(wind_speed, dtype=float)
        ti = np.asarray(turbulence_intensity, dtype=float)
        pitch = np.asarray(platform_pitch_deg, dtype=float)

        rho = self.humid_air_density(air_temperature_degC, relative_humidity)

        # Base power
        p = self._power_spline(v)

        # Turbulence correction (second-order)
        sigma_v = ti * v
        d2p = self._d2p_spline(v)
        p = p + 0.5 * d2p * sigma_v ** 2

        # Air density correction
        p = p * (rho / self.RHO_REF)

        # Platform pitch penalty
        p = p * self.platform_pitch_factor(pitch)

        # Enforce boundaries
        p = np.where(v < self.cut_in, 0.0, p)
        p = np.where(v > self.cut_out, 0.0, p)
        return np.clip(p, 0.0, self.rated_power)

    def power_coefficient(self, wind_speed, turbulence_intensity=0.07,
                          platform_pitch_deg=0.0,
                          air_temperature_degC=12.0, relative_humidity=0.80):
        """Cp = P / (0.5 * rho * A * v^3)."""
        v = np.asarray(wind_speed, dtype=float)
        rho = self.humid_air_density(air_temperature_degC, relative_humidity)
        p_kw = self.power(v, turbulence_intensity, platform_pitch_deg,
                          air_temperature_degC, relative_humidity)
        p_avail = 0.5 * rho * self.rotor_area * v ** 3 / 1000.0
        return np.where(p_avail > 0, p_kw / p_avail, 0.0)
