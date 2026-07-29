"""Standardized predict interface for EC148 F0a bioethanol yield table."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import EthanolYieldTable

_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC148"
    component_name = "Bioethanol Fermentation"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_DATA):
        with open(params_path) as f:
            self.params = json.load(f)
        self.table = EthanolYieldTable(self.params)

    def predict(self, inputs: dict) -> dict:
        feedstock = inputs.get("feedstock", "corn")
        T = float(inputs.get("temperature_degC", self.table.T_opt))
        feed_t = float(inputs.get("feed_tonnes", 1.0))
        y = self.table.ethanol_yield(feedstock, T)
        return {
            "ethanol_L_per_tonne": y,
            "ethanol_L": y * feed_t,
            "energy_yield_MJ_per_tonne": self.table.energy_yield_MJ_tonne(feedstock, T),
            "temp_multiplier": self.table.temp_multiplier(T),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"feedstock": list(self.table.feedstocks),
                       "temperature_degC": "20-45", "feed_tonnes": ">0"},
            "outputs": ["ethanol_L_per_tonne", "ethanol_L", "energy_yield_MJ_per_tonne"],
            "source": self.params["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print(m.predict({"feedstock": "corn", "temperature_degC": 32, "feed_tonnes": 100}))
