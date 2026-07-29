"""F0a empirical model for EC204 Calcium Looping.

Capture rate decays with sorbent cycle number (CaO activity loss) read from a
cycle-number breakpoint curve via np.interp. Specific thermal + electric
energy fixed (calciner duty dominated). Captured CO2 = inlet CO2 * capture
rate at the given cycle.

Source: Abanades et al. (2004) Chem. Eng. J.; Dean et al. (2011).
SEC_thermal=3.2 GJ/tCO2. NumPy only.
"""
import numpy as np


class CaptureDecayCurve:
    def __init__(self, params):
        u = params["unit"]
        self.sec_th = u["SEC_thermal_GJ_tCO2"]["value"]
        self.sec_el = u["SEC_elec_GJ_tCO2"]["value"]
        self.cap0 = u["capture_rate"]["value"]
        self.cyc_pts = np.array(params["decay_curve"]["cycle_number"])
        self.cap_pts = np.array(params["decay_curve"]["capture_rate"])

    def capture_rate(self, cycle):
        return np.interp(np.asarray(cycle, dtype=float), self.cyc_pts, self.cap_pts)

    def predict(self, inputs):
        co2_in = float(inputs.get("co2_in_kg_s", 10.0))
        cycle = float(inputs.get("cycle_number", 1))
        cr = float(self.capture_rate(cycle))
        co2_cap = co2_in * cr
        cap_t_s = co2_cap / 1000.0
        return {
            "co2_captured_kg_s": co2_cap,
            "capture_rate": cr,
            "SEC_thermal_GJ_tCO2": self.sec_th,
            "SEC_elec_GJ_tCO2": self.sec_el,
            "SEC_total_GJ_tCO2": self.sec_th + self.sec_el,
            "thermal_power_MW": self.sec_th * cap_t_s * 1000.0,
        }
