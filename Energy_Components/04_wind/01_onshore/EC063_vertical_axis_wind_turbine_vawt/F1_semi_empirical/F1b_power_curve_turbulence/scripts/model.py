"""
EC063 — Vertical Axis Wind Turbine (VAWT) — F1b Turbulence + Air Density

Extends F1a with:
  1. Air density rho(T, P, altitude) using ISA lapse-rate pressure model.
  2. Turbulence-intensity (TI) effect on effective Cp.

VAWT aerodynamics and TI:
  VAWTs experience the blade at continuously varying angles of attack across
  each rotation. High TI adds unsteady incidence fluctuations on top of this
  inherent cyclic variation, increasing dynamic stall and reducing time-averaged
  Cp. The semi-empirical model applies a linear TI penalty on Cp at partial load
  (v < rated); at rated the turbine is pitched/controlled to maximum output.

      Cp_eff(v, TI) = Cp_base(v) * max(0, 1 + k_ti * TI)

  where k_ti = ti_cp_sensitivity (negative, typically -0.40 for VAWTs).

  Power with TI and density:
      P(v, TI, rho) = P_curve(v) * (rho / rho_ref) * max(0, 1 + k_ti * TI)
  with rated-power clamp applied after.

Air density (ISA model):
    P_alt = P_sea_level * (1 - L*z / T_std)^(g/Rd/L)
    rho = P_alt / (Rd * T_K)

References:
    Tjiu et al. (2015). Renew. Energy 75, 50-67.
    Simão Ferreira et al. (2007). AIAA-2007-1247. (TI effect on VAWT Cp)
    ISO 2533:1975 — Standard Atmosphere.
"""

import numpy as np
from scipy.interpolate import interp1d


class VAWTF1b:
    """Vertical-axis wind turbine — turbulence + air density correction."""

    RHO_REF = 1.225   # kg/m3
    G = 9.80665        # m/s2

    def __init__(self, params: dict):
        t = params["turbine"]
        pc = params["power_curve"]
        atm = params["atmosphere"]

        self.rated_power = t["rated_power"]["value"]          # kW
        self.hub_height = t["hub_height"]["value"]
        self.rotor_diameter = t["rotor_diameter"]["value"]
        self.rotor_height = t["rotor_height"]["value"]
        self.swept_area = t["swept_area"]["value"]
        self.cut_in = t["cut_in_speed"]["value"]
        self.cut_out = t["cut_out_speed"]["value"]
        self.rated_speed = t["rated_speed"]["value"]
        self.k_ti = t["ti_cp_sensitivity"]["value"]           # <0

        self.Rd = atm["Rd"]["value"]
        self.P0 = atm["P_sea_level"]["value"]
        self.L = atm["lapse_rate"]["value"]
        self.T_std = atm["T_std"]["value"]

        self._power_curve = interp1d(
            pc["wind_speeds_ms"], pc["power_kw"],
            kind="linear", bounds_error=False, fill_value=0.0
        )

    # ------------------------------------------------------------------
    # Atmosphere
    # ------------------------------------------------------------------

    def air_density(self, T_degC, altitude_m=0.0):
        """
        Dry air density using ISA pressure model.

        rho = P_alt / (Rd * T_K)
        P_alt = P0 * (1 - L*z/T_std)^(g/Rd/L)
        """
        T_degC = np.asarray(T_degC, dtype=float)
        z = np.asarray(altitude_m, dtype=float)
        T_K = T_degC + 273.15

        exponent = self.G / (self.Rd * self.L)
        P_alt = self.P0 * np.maximum(1.0 - self.L * z / self.T_std, 0.0) ** exponent
        rho = P_alt / (self.Rd * T_K)
        return rho

    # ------------------------------------------------------------------
    # TI modifier
    # ------------------------------------------------------------------

    def ti_modifier(self, wind_speed, turbulence_intensity):
        """
        TI-based Cp/power modifier.

        At rated and above: no penalty (turbine is regulated to rated output).
        Below rated: linear k_ti * TI reduction.
        Clipped at 0 (no negative power from TI alone).
        """
        v = np.asarray(wind_speed, dtype=float)
        ti = np.asarray(turbulence_intensity, dtype=float)

        below_rated = v < self.rated_speed
        raw = 1.0 + self.k_ti * ti
        modifier = np.where(below_rated, raw, 1.0)
        return np.maximum(modifier, 0.0)

    # ------------------------------------------------------------------
    # Power
    # ------------------------------------------------------------------

    def power(self, wind_speed, turbulence_intensity=0.0,
              air_temperature_degC=15.0, altitude_m=0.0):
        """
        Power output in kW with density and TI corrections.

        Args:
            wind_speed: m/s (scalar or array)
            turbulence_intensity: TI 0–0.50
            air_temperature_degC: degC
            altitude_m: site altitude above sea level, m
        """
        v = np.asarray(wind_speed, dtype=float)
        ti = np.asarray(turbulence_intensity, dtype=float)

        rho = self.air_density(air_temperature_degC, altitude_m)
        p = self._power_curve(v)
        p = p * (rho / self.RHO_REF)
        p = p * self.ti_modifier(v, ti)
        p = np.where(v < self.cut_in, 0.0, p)
        p = np.where(v > self.cut_out, 0.0, p)
        return np.clip(p, 0.0, self.rated_power)

    def power_coefficient(self, wind_speed, turbulence_intensity=0.0,
                          air_temperature_degC=15.0, altitude_m=0.0):
        """Cp = P / (0.5 * rho * A * v^3)."""
        v = np.asarray(wind_speed, dtype=float)
        rho = self.air_density(air_temperature_degC, altitude_m)
        p_kw = self.power(v, turbulence_intensity, air_temperature_degC, altitude_m)
        p_avail = 0.5 * rho * self.swept_area * v ** 3 / 1000.0
        return np.where(p_avail > 0, p_kw / p_avail, 0.0)

    def capacity_factor(self, wind_speed, turbulence_intensity=0.0,
                        air_temperature_degC=15.0, altitude_m=0.0):
        """Instantaneous capacity factor = P / P_rated."""
        return self.power(wind_speed, turbulence_intensity,
                          air_temperature_degC, altitude_m) / self.rated_power
