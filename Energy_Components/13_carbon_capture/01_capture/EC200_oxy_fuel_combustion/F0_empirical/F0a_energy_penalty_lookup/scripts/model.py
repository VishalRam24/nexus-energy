"""F0a empirical model for EC200 Oxy-Fuel Combustion Capture.

Algebraic O2 demand from coal carbon/hydrogen/sulfur content, then ASU +
CO2-compression energy penalty (kWh/tCO2 -> GJ/tCO2). Capture rate is read
from a load-dependent breakpoint curve (np.interp): near-complete at full
load, slightly lower at part load.

Source: Buhre et al. (2005) Prog. Energy Combust. Sci. 31(4) 283-307;
Toftegaard et al. (2010) 36(5) 581-625. NumPy only.
"""
import numpy as np


class EnergyPenaltyLookup:
    def __init__(self, params):
        u = params["unit"]
        self.fC = u["fuel_C"]["value"]
        self.fH = u["fuel_H"]["value"]
        self.fS = u["fuel_S"]["value"]
        self.asu = u["asu_specific_energy"]["value"]          # kWh/tO2
        self.comp = u["compression_specific_energy"]["value"]  # kWh/tCO2
        self.o2_purity = u["o2_purity"]["value"]
        self.excess = u["excess_o2_ratio"]["value"]
        self.load_pts = np.array(params["capture_curve"]["load"])
        self.cap_pts = np.array(params["capture_curve"]["capture_rate"])

    def capture_rate(self, load):
        return np.interp(np.asarray(load, dtype=float), self.load_pts, self.cap_pts)

    def predict(self, inputs):
        fuel = float(inputs.get("fuel_rate", 10.0))   # kg/s coal
        load = float(inputs.get("load", 1.0))
        # stoichiometric O2 (kg) per kg fuel: C+O2, H4->2H2O (O2/H stoich), S+O2
        o2_per_fuel = (self.fC * 31.998 / 12.011
                       + self.fH * (31.998 / 4.0) / 1.008
                       + self.fS * 31.998 / 32.06)
        o2_dem = fuel * o2_per_fuel * (1.0 + self.excess) / self.o2_purity  # kg/s
        co2_gen = fuel * self.fC * 44.01 / 12.011                            # kg/s CO2
        cr = float(self.capture_rate(load))
        co2_cap = co2_gen * cr
        # energy: ASU kWh/tO2 * tO2/s + comp kWh/tCO2 * tCO2/s -> kW -> MW
        asu_MW = self.asu * (o2_dem / 1000.0)
        comp_MW = self.comp * (co2_cap / 1000.0)
        # specific penalty per tCO2 captured
        sec_kWh_t = (asu_MW + comp_MW) * 1000.0 / max(co2_cap, 1e-9)
        return {
            "o2_demand_kg_s": o2_dem,
            "co2_generated_kg_s": co2_gen,
            "co2_captured_kg_s": co2_cap,
            "capture_rate": cr,
            "asu_power_MW": asu_MW,
            "compression_power_MW": comp_MW,
            "specific_penalty_kWh_tCO2": sec_kWh_t,
        }
