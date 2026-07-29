"""Standardized predict interface for EC147 F0a HTL bio-crude yield table."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import BiocrudeTable

_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC147"
    component_name = "Hydrothermal Liquefaction (HTL)"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_DATA):
        with open(params_path) as f:
            self.params = json.load(f)
        self.table = BiocrudeTable(self.params)

    def predict(self, inputs: dict) -> dict:
        feedstock = inputs.get("feedstock", "microalgae_chlorella")
        T = float(inputs.get("temperature_degC", self.table.T_ref))
        feed_kg = float(inputs.get("dry_feed_kg", 1.0))
        y = self.table.biocrude_yield(feedstock, T)
        return {
            "biocrude_yield": y,
            "biocrude_kg": y * feed_kg,
            "energy_yield_MJ_kg": self.table.energy_yield_MJ_kg(feedstock, T),
            "HHV_biocrude_MJ_kg": self.table.hhv,
            "temp_multiplier": self.table.temp_multiplier(T),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"feedstock": list(self.table.feedstocks),
                       "temperature_degC": "250-400", "dry_feed_kg": ">0"},
            "outputs": ["biocrude_yield", "biocrude_kg", "energy_yield_MJ_kg"],
            "source": self.params["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print(m.predict({"feedstock": "microalgae_nannochloropsis", "temperature_degC": 330, "dry_feed_kg": 1000}))
