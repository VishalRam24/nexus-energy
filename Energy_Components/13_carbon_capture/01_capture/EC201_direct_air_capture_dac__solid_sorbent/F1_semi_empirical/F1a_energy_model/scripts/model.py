"""
EC201 — Direct Air Capture (DAC) Solid Sorbent — F1a Energy Model

Energy consumption model for solid-sorbent DAC systems:
  - E_thermal = E_th_base / (capture_efficiency * humidity_factor)
  - E_electric = E_el_base  (fans + vacuum, relatively constant)
  - humidity_factor = 1 + 0.3 * (RH - 0.5)   [RH in 0-1]
  - CO2_concentration_mass = 415e-6 * (44/29) * rho_air  [kg_CO2/m3]
  - capture_rate = air_flow * CO2_conc_mass * capture_efficiency

Reference:
    Fasihi et al. (2019). J. Cleaner Production, 224, 957-980.
"""

import numpy as np


class DACF1a:
    """Solid-sorbent DAC energy model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.E_th_base = u["E_th_base"]["value"]          # kWh_th/tCO2
        self.E_el_base = u["E_el_base"]["value"]           # kWh_e/tCO2
        self.capture_efficiency = u["capture_efficiency"]["value"]
        self.CO2_ppm = u["CO2_concentration_ppm"]["value"]  # ppm_v
        self.rho_air = u["rho_air"]["value"]               # kg/m3
        self.M_CO2 = u["CO2_molar_mass"]["value"]          # g/mol
        self.M_air = u["air_molar_mass"]["value"]          # g/mol
        # Mass fraction of CO2 in air -> kg_CO2 / m3_air
        self.CO2_conc_kgm3 = (self.CO2_ppm * 1e-6) * (self.M_CO2 / self.M_air) * self.rho_air

    def humidity_factor(self, relative_humidity):
        """Factor >1 when dry (low RH) — sorbent performance drops at very low humidity.
        At RH=0.5 (reference): factor=1.0; higher RH slightly improves performance."""
        rh = np.asarray(relative_humidity, dtype=float)
        return 1.0 + 0.3 * (rh - 0.5)

    def specific_thermal(self, relative_humidity):
        """Specific thermal energy consumption [kWh_th/tCO2]."""
        hf = self.humidity_factor(relative_humidity)
        return self.E_th_base / (self.capture_efficiency * hf)

    def specific_electric(self):
        """Specific electrical energy consumption [kWh_e/tCO2] — constant."""
        return float(self.E_el_base)

    def capture_rate_kg_s(self, air_flow_m3h):
        """CO2 capture rate [kg/s] at rated capture efficiency."""
        flow_m3s = np.asarray(air_flow_m3h, dtype=float) / 3600.0
        return flow_m3s * self.CO2_conc_kgm3 * self.capture_efficiency

    def annual_outputs(self, air_flow_m3h, relative_humidity, ambient_temp=15.0):
        """Compute annual energy and capture metrics.

        Parameters
        ----------
        air_flow_m3h : float or array
            Air throughput [m3/hr]
        relative_humidity : float or array
            Relative humidity [0–1]
        ambient_temp : float or array
            Ambient temperature [degC] — reserved for future corrections

        Returns
        -------
        dict with co2_captured_tpa, thermal_energy_mwh_pa, electrical_energy_mwh_pa,
             specific_thermal_kwht, specific_electric_kwhe
        """
        air_flow = np.asarray(air_flow_m3h, dtype=float)
        rh = np.asarray(relative_humidity, dtype=float)

        # Annual CO2 captured [tonnes/year]
        rate_kg_s = self.capture_rate_kg_s(air_flow)
        co2_tpa = rate_kg_s * 3600.0 * 8760.0 / 1000.0  # kg/s -> t/yr

        # Specific energies
        e_th = self.specific_thermal(rh)    # kWh_th/tCO2
        e_el = self.specific_electric()     # kWh_e/tCO2

        # Annual energies [MWh/year]
        thermal_mwh_pa = co2_tpa * e_th / 1000.0
        electric_mwh_pa = co2_tpa * e_el / 1000.0

        return {
            "co2_captured_tpa": co2_tpa,
            "thermal_energy_mwh_pa": thermal_mwh_pa,
            "electrical_energy_mwh_pa": electric_mwh_pa,
            "specific_thermal_kwht": e_th,
            "specific_electric_kwhe": np.full_like(co2_tpa, e_el),
        }
