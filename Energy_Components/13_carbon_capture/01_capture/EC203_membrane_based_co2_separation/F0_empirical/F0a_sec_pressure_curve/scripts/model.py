"""F0a empirical model for EC203 Membrane-Based CO2 Separation.

Specific energy consumption (MJ/kgCO2) read from a pressure-ratio breakpoint
curve via np.interp (SEC rises with pressure ratio). CO2 recovery and permeate
purity fixed at design. Captured CO2 = feed CO2 * recovery.

Source: Baker (2002) Ind. Eng. Chem. Res.; Merkel et al. (2010).
NumPy only.
"""
import numpy as np


class SECPressureCurve:
    def __init__(self, params):
        u = params["unit"]
        self.recovery = u["CO2_recovery"]["value"]
        self.purity = u["CO2_purity"]["value"]
        self.pr_pts = np.array(params["sec_curve"]["pressure_ratio"])
        self.sec_pts = np.array(params["sec_curve"]["SEC_MJ_kgCO2"])

    def sec(self, pr):
        return np.interp(np.asarray(pr, dtype=float), self.pr_pts, self.sec_pts)

    def predict(self, inputs):
        co2_feed = float(inputs.get("co2_feed_kg_s", 10.0))
        pr = float(inputs.get("pressure_ratio", 10.0))
        rec = float(inputs.get("recovery", self.recovery))
        co2_cap = co2_feed * rec
        sec = float(self.sec(pr))                     # MJ/kgCO2
        power_MW = sec * co2_cap                       # MJ/kg * kg/s = MW
        return {
            "co2_captured_kg_s": co2_cap,
            "recovery": rec,
            "permeate_purity": self.purity,
            "SEC_MJ_kgCO2": sec,
            "SEC_GJ_tCO2": sec,  # MJ/kg == GJ/t numerically
            "compression_power_MW": power_MW,
        }
