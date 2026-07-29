"""
EC111 -- Diesel Generator -- F1b Part-Load + Ambient (Altitude & Temperature Derating)

Extends F1a Willans-line model by adding:
  1. Altitude derating: 3.5% power loss per 300m above 1000m ASL
  2. Temperature derating: 0.5% power loss per degC above 40 degC
  3. Exhaust temperature model

Willans line (fuel consumption linear with electrical output):
    fuel_rate [L/h] = a + b * P_elec [kW]

Derating:
    f_alt  = max(0.5, 1 - 0.035/300 * max(0, altitude - 1000))
    f_temp = max(0.5, 1 - 0.005 * max(0, T_amb - 40))
    P_rated_derated = P_rated * f_alt * f_temp

Part-load efficiency (derived from Willans line):
    eta = P_elec / (fuel_rate * rho * LHV / 3.6)

SFC [g/kWh] = fuel_rate * rho * 1000 / P_elec

References:
    US Army TM 5-811-6 (1996). Electric Power Plant Design.
    Caterpillar Application and Installation Guide (2017).
    ISO 8528-1:2018 Reciprocating internal combustion engine driven alternating
        current generating sets -- Part 1.
"""

import numpy as np


class DieselGeneratorF1b:
    """Diesel generator with Willans line, altitude and temperature derating."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated        = u["rated_power_kw"]["value"]                # kW
        self.a_willans      = u["a_willans_lph"]["value"]                 # L/h
        self.b_willans      = u["b_willans_lkwh"]["value"]                # L/kWh
        self.rho            = u["rho_diesel_kgl"]["value"]                # kg/L
        self.LHV            = u["LHV_diesel_mj_kg"]["value"]              # MJ/kg
        self.PLR_min        = u["PLR_min"]["value"]
        self.alt_derate_pct = u["altitude_derating_pct_per_300m"]["value"]  # %
        self.alt_start      = u["altitude_derating_start_m"]["value"]       # m
        self.temp_derate_pct = u["temp_derating_pct_per_degC"]["value"]    # %/degC
        self.temp_start     = u["temp_derating_start_c"]["value"]           # degC
        self.co2_factor     = u["co2_factor_kgl"]["value"]
        self.T_exh_rated    = u["T_exhaust_rated_c"]["value"]              # degC
        self.T_exh_idle     = u["T_exhaust_idle_c"]["value"]               # degC

    # ------------------------------------------------------------------
    # Derating
    # ------------------------------------------------------------------

    def f_altitude(self, altitude_m):
        """Altitude derating factor: 3.5% per 300m above 1000m."""
        alt = np.asarray(altitude_m, dtype=float)
        excess = np.maximum(0.0, alt - self.alt_start)
        factor = 1.0 - (self.alt_derate_pct / 100.0) / 300.0 * excess
        return np.clip(factor, 0.5, 1.0)

    def f_temperature(self, T_amb_c):
        """Temperature derating factor: 0.5% per degC above 40 degC."""
        T = np.asarray(T_amb_c, dtype=float)
        excess = np.maximum(0.0, T - self.temp_start)
        factor = 1.0 - (self.temp_derate_pct / 100.0) * excess
        return np.clip(factor, 0.5, 1.0)

    def rated_power_derated(self, T_amb_c, altitude_m):
        """Derated rated power [kW]."""
        return self.P_rated * self.f_altitude(altitude_m) * self.f_temperature(T_amb_c)

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def power_output_kw(self, PLR, T_amb_c=25.0, altitude_m=0.0):
        """Electrical output [kW] = PLR * P_rated_derated."""
        PLR = np.asarray(PLR, dtype=float)
        P_rated_eff = self.rated_power_derated(T_amb_c, altitude_m)
        return PLR * P_rated_eff

    def fuel_consumption_l_h(self, PLR, T_amb_c=25.0, altitude_m=0.0):
        """Fuel consumption [L/h] from Willans line: a + b * P_elec."""
        P_elec = self.power_output_kw(PLR, T_amb_c, altitude_m)
        return self.a_willans + self.b_willans * P_elec

    def efficiency(self, PLR, T_amb_c=25.0, altitude_m=0.0):
        """Generator efficiency = P_elec / (fuel_energy_rate).
        fuel_energy_rate [kW] = fuel_rate [L/h] * rho [kg/L] * LHV [MJ/kg] * 1e3/3.6
        """
        P_elec = self.power_output_kw(PLR, T_amb_c, altitude_m)
        fuel_lh = self.fuel_consumption_l_h(PLR, T_amb_c, altitude_m)
        fuel_kw = fuel_lh * self.rho * self.LHV * 1e3 / 3600.0  # MJ/kg * kg/L * L/h -> kW
        eta = np.where(
            P_elec > 0,
            np.clip(P_elec / fuel_kw, 0.0, 0.50),
            0.0
        )
        return eta

    def sfc_g_kwh(self, PLR, T_amb_c=25.0, altitude_m=0.0):
        """Specific fuel consumption [g/kWh]."""
        P_elec = self.power_output_kw(PLR, T_amb_c, altitude_m)
        fuel_lh = self.fuel_consumption_l_h(PLR, T_amb_c, altitude_m)
        sfc = np.where(
            P_elec > 0,
            fuel_lh * self.rho * 1000.0 / P_elec,
            np.nan
        )
        return sfc

    def exhaust_temp_c(self, PLR):
        """Exhaust temperature [degC] -- linear interpolation between idle and rated."""
        PLR = np.asarray(PLR, dtype=float)
        return self.T_exh_idle + (self.T_exh_rated - self.T_exh_idle) * PLR
