"""F0a empirical energy model for EC201 DAC (solid sorbent).

Specific thermal + electrical energy per tonne CO2 captured. Thermal energy
varies mildly with ambient temperature via an np.interp breakpoint curve
(colder air -> slightly higher regeneration demand). Captured CO2 from air
throughput, ambient CO2 ppm and capture efficiency.

Source: Fasihi et al. (2019), J. Cleaner Production 224, 957-980
(~1500 kWh_th + 250 kWh_e per tCO2). NumPy only.
"""
import numpy as np


class EnergyLookup:
    def __init__(self, params):
        u = params["unit"]
        self.E_el = u["E_el_base"]["value"]                # kWh_e/tCO2
        self.cap_eff = u["capture_efficiency"]["value"]
        self.ppm = u["CO2_concentration_ppm"]["value"]
        self.rho_air = u["rho_air"]["value"]
        self.mw_co2 = u["CO2_molar_mass"]["value"]
        self.mw_air = u["air_molar_mass"]["value"]
        self.T_pts = np.array(params["thermal_curve"]["ambient_C"])
        self.Eth_pts = np.array(params["thermal_curve"]["E_th_kWh_tCO2"])

    def thermal_energy(self, T_C):
        return np.interp(np.asarray(T_C, dtype=float), self.T_pts, self.Eth_pts)

    def predict(self, inputs):
        air = float(inputs.get("air_flow_m3h", 1.0e6))     # m3/hr
        T = float(inputs.get("ambient_temp", 25.0))
        # CO2 mass available: air mass * (ppm volume frac * MW ratio)
        air_kg_h = air * self.rho_air
        co2_vol = self.ppm * 1e-6
        co2_mass_frac = co2_vol * self.mw_co2 / (
            co2_vol * self.mw_co2 + (1 - co2_vol) * self.mw_air)
        co2_in_kg_h = air_kg_h * co2_mass_frac
        co2_cap_kg_h = co2_in_kg_h * self.cap_eff
        E_th = float(self.thermal_energy(T))               # kWh_th/tCO2
        cap_t_h = co2_cap_kg_h / 1000.0
        return {
            "co2_captured_kg_h": co2_cap_kg_h,
            "capture_efficiency": self.cap_eff,
            "E_thermal_kWh_tCO2": E_th,
            "E_electric_kWh_tCO2": self.E_el,
            "total_energy_kWh_tCO2": E_th + self.E_el,
            "thermal_power_kW": E_th * cap_t_h,
            "electric_power_kW": self.E_el * cap_t_h,
        }
