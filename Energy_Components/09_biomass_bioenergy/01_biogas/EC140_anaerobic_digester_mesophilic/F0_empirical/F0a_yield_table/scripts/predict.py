"""Standardized predict interface for EC140 F0a empirical yield lookup."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import YieldTable

_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC140"
    component_name = "Anaerobic Digester (Mesophilic)"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_DATA):
        with open(params_path) as f:
            self.params = json.load(f)
        self.table = YieldTable(self.params)

    def predict(self, inputs: dict) -> dict:
        feedstock = inputs.get("feedstock", "cattle_manure")
        T = float(inputs.get("temperature_degC", self.table.T_opt))
        vs_kg = float(inputs.get("vs_fed_kg", 1.0))
        ch4 = self.table.methane_yield(feedstock, T)
        energy = self.table.energy_yield(feedstock, T)
        return {
            "ch4_yield_m3_kgVS": ch4,
            "energy_yield_kwh_kgVS": energy,
            "ch4_total_m3": ch4 * vs_kg,
            "energy_total_kwh": energy * vs_kg,
            "methane_fraction": self.table.ch4_frac.get(feedstock, 0.60),
            "temp_multiplier": self.table.temp_multiplier(T),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"feedstock": list(self.table.feedstocks),
                       "temperature_degC": "25-55", "vs_fed_kg": ">0"},
            "outputs": ["ch4_yield_m3_kgVS", "energy_yield_kwh_kgVS",
                        "ch4_total_m3", "energy_total_kwh"],
            "source": self.params["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print(m.predict({"feedstock": "food_waste", "temperature_degC": 37, "vs_fed_kg": 1000}))
