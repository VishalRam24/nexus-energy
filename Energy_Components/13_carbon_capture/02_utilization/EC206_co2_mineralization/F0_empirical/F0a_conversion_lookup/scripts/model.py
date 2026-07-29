"""F0a empirical model for EC206 CO2 Mineralization.

Conversion (fraction permanently mineralized) read from a temperature
breakpoint curve via np.interp (carbonation improves with temperature up to
an optimum). Fixed specific energy (grinding/pumping) and carbonate yield.

Source: Sanna et al. (2014) Prog. Energy Combust. Sci.; IPCC (2005) CCS.
conversion ~0.80, SEC ~0.5 GJ/tCO2. NumPy only.
"""
import numpy as np


class ConversionLookup:
    def __init__(self, params):
        u = params["unit"]
        self.sec = u["SEC_GJ_tCO2"]["value"]
        self.yield_kg = u["carbonate_yield_kg_kg"]["value"]
        self.conv0 = u["conversion"]["value"]
        self.T_pts = np.array(params["conversion_curve"]["T_C"])
        self.conv_pts = np.array(params["conversion_curve"]["conversion"])

    def conversion(self, T_C):
        return np.interp(np.asarray(T_C, dtype=float), self.T_pts, self.conv_pts)

    def predict(self, inputs):
        co2_in = float(inputs.get("co2_in_kg_s", 1.0))
        T = float(inputs.get("temperature_C", 100.0))
        conv = float(self.conversion(T))
        co2_min = co2_in * conv
        carbonate = co2_min * self.yield_kg
        power_MW = self.sec * (co2_min / 1000.0) * 1000.0   # GJ/t * t/s *1e3 = MW
        return {
            "conversion": conv,
            "co2_mineralized_kg_s": co2_min,
            "carbonate_produced_kg_s": carbonate,
            "SEC_GJ_tCO2": self.sec,
            "electric_power_MW": power_MW,
        }
