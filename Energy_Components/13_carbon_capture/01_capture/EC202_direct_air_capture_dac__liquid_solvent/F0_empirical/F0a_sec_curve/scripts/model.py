"""F0a empirical model for EC202 DAC (liquid solvent, KOH/calciner).

Capture rate fixed at design; specific thermal + electric energy (GJ/tCO2)
with a mild part-load penalty read from a capacity-fraction breakpoint curve
(np.interp): SEC rises as load drops (calciner standby losses).

Source: Keith et al. (2018) Carbon Engineering; Fasihi et al. (2019).
~6.0 GJ_th + ~1.8 GJ_e per tCO2 at full load. NumPy only.
"""
import numpy as np


class SECCurve:
    def __init__(self, params):
        u = params["unit"]
        self.cap = u["capture_rate"]["value"]
        self.sec_th = u["SEC_thermal_GJ_tCO2"]["value"]
        self.sec_el = u["SEC_elec_GJ_tCO2"]["value"]
        self.load_pts = np.array(params["partload_curve"]["capacity_fraction"])
        self.mult_pts = np.array(params["partload_curve"]["sec_multiplier"])

    def partload_multiplier(self, frac):
        return np.interp(np.asarray(frac, dtype=float), self.load_pts, self.mult_pts)

    def predict(self, inputs):
        co2_rated = float(inputs.get("co2_rated_kg_s", 1.0))
        frac = float(inputs.get("capacity_fraction", 1.0))
        mult = float(self.partload_multiplier(frac))
        co2_cap = co2_rated * frac
        sec_th = self.sec_th * mult
        sec_el = self.sec_el * mult
        cap_t_s = co2_cap / 1000.0
        return {
            "co2_captured_kg_s": co2_cap,
            "capture_rate": self.cap,
            "SEC_thermal_GJ_tCO2": sec_th,
            "SEC_elec_GJ_tCO2": sec_el,
            "SEC_total_GJ_tCO2": sec_th + sec_el,
            "thermal_power_MW": sec_th * cap_t_s * 1000.0,
            "electric_power_MW": sec_el * cap_t_s * 1000.0,
        }
