"""F0a empirical model for EC205 CO2 Electrolyzer (CO2 -> CO).

Faradaic efficiency and specific energy (kWh/kgCO2) read from a current-
density breakpoint curve via np.interp (FE drops, SEC rises at high current
density). CO produced from converted CO2.

Source: Jouny et al. (2018) Ind. Eng. Chem. Res.; Bushuyev et al. (2018) Joule.
SEC ~8 kWh/kgCO2, FE ~0.85, V_cell ~3.0 V. NumPy only.
"""
import numpy as np


class SECCurrentCurve:
    def __init__(self, params):
        u = params["unit"]
        self.V = u["V_cell"]["value"]
        self.cd_pts = np.array(params["operating_curve"]["current_density_mA_cm2"])
        self.fe_pts = np.array(params["operating_curve"]["faradaic_efficiency"])
        self.sec_pts = np.array(params["operating_curve"]["SEC_kWh_kgCO2"])

    def _interp(self, x, xp, fp):
        return float(np.interp(float(x), xp, fp))

    def predict(self, inputs):
        co2_in = float(inputs.get("co2_in_kg_s", 1.0))
        cd = float(inputs.get("current_density_mA_cm2", 200.0))
        fe = self._interp(cd, self.cd_pts, self.fe_pts)
        sec = self._interp(cd, self.cd_pts, self.sec_pts)   # kWh/kgCO2
        co2_conv = co2_in * fe
        # CO produced: 44 g CO2 -> 28 g CO
        co_out = co2_conv * (28.01 / 44.01)
        power_kW = sec * co2_conv                            # kWh/kg * kg/s = kW
        return {
            "faradaic_efficiency": fe,
            "co2_converted_kg_s": co2_conv,
            "co_produced_kg_s": co_out,
            "SEC_kWh_kgCO2": sec,
            "cell_voltage_V": self.V,
            "power_kW": power_kW,
        }
