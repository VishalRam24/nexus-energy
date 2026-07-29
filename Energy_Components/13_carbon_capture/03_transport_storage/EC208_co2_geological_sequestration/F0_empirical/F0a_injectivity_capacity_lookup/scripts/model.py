"""F0a empirical model for EC208 CO2 Geological Sequestration.

Injection rate read from a wellhead-overpressure breakpoint curve (np.interp):
injectivity ~proportional to (P_wellhead - P_reservoir). Static storage
capacity from pore volume: A * h * porosity * storage_efficiency * rho_CO2.

Source: van der Meer (1993); IPCC (2005) CCS Ch.5; Benson & Cole (2008).
NumPy only.
"""
import numpy as np


class InjectivityCapacityLookup:
    def __init__(self, params):
        r = params["reservoir"]
        c = params["co2"]
        self.P_res = r["P_reservoir_bar"]["value"]
        self.thickness = r["thickness_m"]["value"]
        self.area_km2 = r["area_km2"]["value"]
        self.porosity = r["porosity"]["value"]
        self.eff = r["storage_efficiency"]["value"]
        self.rho = c["rho_injection"]["value"]
        self.dP_pts = np.array(params["injectivity_curve"]["overpressure_bar"])
        self.q_pts = np.array(params["injectivity_curve"]["injection_kg_s"])

    def injection_rate(self, overpressure):
        return np.interp(np.asarray(overpressure, dtype=float), self.dP_pts, self.q_pts)

    def storage_capacity_Mt(self, area_km2=None):
        a = self.area_km2 if area_km2 is None else area_km2
        pore_m3 = (a * 1e6) * self.thickness * self.porosity
        co2_kg = pore_m3 * self.eff * self.rho
        return co2_kg / 1e9  # Mt

    def predict(self, inputs):
        P_wh = float(inputs.get("P_wellhead_bar", 150.0))
        area = float(inputs.get("area_km2", self.area_km2))
        overpressure = max(P_wh - self.P_res, 0.0)
        q = float(self.injection_rate(overpressure))
        cap = self.storage_capacity_Mt(area)
        return {
            "overpressure_bar": overpressure,
            "injection_rate_kg_s": q,
            "injection_rate_Mt_yr": q * 3.1536e7 / 1e9,
            "storage_capacity_Mt": cap,
        }
