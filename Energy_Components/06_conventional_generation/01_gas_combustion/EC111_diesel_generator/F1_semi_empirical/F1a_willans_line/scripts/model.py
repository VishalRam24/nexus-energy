"""
EC111 — Diesel Generator — F1a Willans Line Model

Willans line: fuel consumption is linear with power output
    fuel_rate = a + b * P_out      [L/h]
    eta       = P_out / (fuel_rate * rho * LHV / 3.6)   [dimensionless]
    SFC       = fuel_rate * rho_diesel * 1000 / P_out    [g/kWh]
    CO2       = fuel_rate * co2_factor                   [kg_CO2/h]

References:
    US Army TM 5-811-6 (1996). Electric Power Plant Design.
    Tuffaha & Gravdahl (2014). Modeling and control of diesel generators in
        a microgrid. IEEE MELECON.
"""

import numpy as np


class DieselGeneratorF1a:
    """Diesel generator — Willans line semi-empirical model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["P_rated_kw"]["value"]           # kW
        self.a = u["a_no_load_lph"]["value"]               # L/h
        self.b = u["b_incremental_lkwh"]["value"]          # L/kWh
        self.rho = u["rho_diesel_kgl"]["value"]            # kg/L
        self.LHV = u["LHV_diesel_mjkg"]["value"]           # MJ/kg
        self.PLR_min = u["PLR_min"]["value"]               # dimensionless
        self.co2_factor = u["co2_factor_kgl"]["value"]     # kg_CO2/L
        self.derating = u["derating_factor_per_degC"]["value"]  # 1/degC
        self.T_ref = u["T_ref_c"]["value"]                 # degC

    def _derate(self, P_out, ambient_temp_c):
        """Derate rated power for ambient temperature above T_ref."""
        T = np.asarray(ambient_temp_c, dtype=float)
        factor = np.clip(1.0 - self.derating * np.maximum(0.0, T - self.T_ref), 0.5, 1.0)
        P_rated_eff = self.P_rated * factor
        return P_rated_eff

    def fuel_rate(self, power_output_kw, ambient_temp_c=25.0):
        """
        Fuel consumption rate [L/h].
        At no-load (P=0): returns a (idle fuel). Below PLR_min: returns a (idling).
        """
        P = np.asarray(power_output_kw, dtype=float)
        P_rated_eff = self._derate(P, ambient_temp_c)
        P_min = self.PLR_min * P_rated_eff
        # Below minimum load → idle fuel consumption
        P_actual = np.where(P < P_min, 0.0, np.clip(P, 0.0, P_rated_eff))
        fr = self.a + self.b * P_actual
        return fr

    def efficiency(self, power_output_kw, ambient_temp_c=25.0):
        """
        Generator efficiency = P_out / (fuel_energy_rate).
        fuel_energy_rate [kW] = fuel_rate [L/h] * rho [kg/L] * LHV [MJ/kg] * 1e6/3600
        Returns 0 when P_out=0 (no-load).
        """
        P = np.asarray(power_output_kw, dtype=float)
        fr = self.fuel_rate(P, ambient_temp_c)
        # fuel energy rate in kW
        fuel_kw = fr * self.rho * self.LHV * 1e6 / 3600.0
        eta = np.where(
            (P > 0) & (fuel_kw > 0),
            np.clip(P / fuel_kw, 0.0, 0.45),
            0.0
        )
        return eta

    def sfc(self, power_output_kw, ambient_temp_c=25.0):
        """
        Specific Fuel Consumption [g/kWh].
        SFC = fuel_rate [L/h] * rho [kg/L] * 1000 [g/kg] / P_out [kW]
        Returns NaN at zero load.
        """
        P = np.asarray(power_output_kw, dtype=float)
        fr = self.fuel_rate(P, ambient_temp_c)
        sfc = np.where(
            P > 0,
            fr * self.rho * 1000.0 / P,
            np.nan
        )
        return sfc

    def co2_emissions(self, power_output_kw, ambient_temp_c=25.0):
        """CO2 emission rate [kg_CO2/h]."""
        fr = self.fuel_rate(power_output_kw, ambient_temp_c)
        return fr * self.co2_factor
