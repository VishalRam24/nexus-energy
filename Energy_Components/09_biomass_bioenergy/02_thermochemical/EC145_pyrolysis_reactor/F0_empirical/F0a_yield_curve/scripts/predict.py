"""Standardized predict interface for EC145 F0a pyrolysis yield curve."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import YieldCurve

_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC145"
    component_name = "Pyrolysis Reactor"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_DATA):
        with open(params_path) as f:
            self.params = json.load(f)
        self.curve = YieldCurve(self.params)

    def predict(self, inputs: dict) -> dict:
        T = float(inputs.get("temperature_degC", self.curve.T_peak))
        feed_kg = float(inputs.get("feed_kg", 1.0))
        y = self.curve.yields(T)
        return {
            "bio_oil_frac": y["bio_oil"],
            "char_frac": y["char"],
            "gas_frac": y["gas"],
            "bio_oil_kg": y["bio_oil"] * feed_kg,
            "char_kg": y["char"] * feed_kg,
            "gas_kg": y["gas"] * feed_kg,
            "product_energy_MJ_kg": self.curve.energy_density_MJ_kg(T),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"temperature_degC": "300-700", "feed_kg": ">0"},
            "outputs": ["bio_oil_frac", "char_frac", "gas_frac",
                        "bio_oil_kg", "product_energy_MJ_kg"],
            "source": self.params["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print(m.predict({"temperature_degC": 500, "feed_kg": 1000}))
