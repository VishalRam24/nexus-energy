"""F0a empirical model for EC207 CO2 Compression & Pipeline.

Compression specific energy (kWh/tCO2) read from an outlet-pressure breakpoint
curve via np.interp (more energy to reach higher pipeline pressure). Pipeline
pressure drop approximated as a linear length term. Dense-phase CO2.

Source: IPCC (2005) CCS Special Report Ch.4; McCoy & Rubin (2008) EES.
~100 kWh/tCO2 for 1 -> 150 bar. NumPy only.
"""
import numpy as np


class CompressionEnergyCurve:
    def __init__(self, params):
        u = params["co2"]
        self.rho = u["rho_dense_phase"]["value"]
        self.dp_per_km = params["pipeline"]["dp_bar_per_km"]["value"]
        self.P_pts = np.array(params["compression_curve"]["P_outlet_bar"])
        self.sec_pts = np.array(params["compression_curve"]["SEC_kWh_tCO2"])

    def sec(self, P_out):
        return np.interp(np.asarray(P_out, dtype=float), self.P_pts, self.sec_pts)

    def predict(self, inputs):
        mdot = float(inputs.get("mass_flow", 100.0))         # kg/s
        P_out = float(inputs.get("P_outlet", 150.0))
        length = float(inputs.get("pipeline_length_km", 100.0))
        sec = float(self.sec(P_out))                          # kWh/tCO2
        power_MW = sec * (mdot / 1000.0)                      # kWh/t * t/s = kW -> MW? kW
        power_MW = power_MW / 1000.0                          # kW -> MW
        dp = self.dp_per_km * length                          # bar over pipeline
        return {
            "SEC_kWh_tCO2": sec,
            "compression_power_MW": power_MW,
            "pipeline_pressure_drop_bar": dp,
            "outlet_pressure_bar": P_out,
            "dense_phase_density_kg_m3": self.rho,
        }
