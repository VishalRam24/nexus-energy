"""Standardized predict interface for EC146 F0a torrefaction mass-yield table."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import MassYieldTable

_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC146"
    component_name = "Torrefaction Reactor"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_DATA):
        with open(params_path) as f:
            self.params = json.load(f)
        self.table = MassYieldTable(self.params)

    def predict(self, inputs: dict) -> dict:
        T = float(inputs.get("temperature_degC", 280.0))
        feed_kg = float(inputs.get("feed_kg", 1.0))
        my = self.table.mass_yield(T)
        return {
            "mass_yield": my,
            "EDR": self.table.edr(T),
            "energy_yield": self.table.energy_yield(T),
            "torrefied_LHV_MJ_kg": self.table.torrefied_lhv(T),
            "solid_kg": my * feed_kg,
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"temperature_degC": "200-300", "feed_kg": ">0"},
            "outputs": ["mass_yield", "EDR", "energy_yield",
                        "torrefied_LHV_MJ_kg", "solid_kg"],
            "source": self.params["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print(m.predict({"temperature_degC": 280, "feed_kg": 1000}))
