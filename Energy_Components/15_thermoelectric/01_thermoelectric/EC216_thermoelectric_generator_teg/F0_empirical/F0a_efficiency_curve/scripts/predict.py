"""EC216 TEG F0a - standard prediction interface."""
import json
import os
import numpy as np

from model import TEGEfficiencyCurve


class ComponentModel:
    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        self.params_path = params_path
        with open(params_path) as f:
            self._p = json.load(f)
        self.component_id = "EC216"
        self.component_name = "Thermoelectric Generator (TEG)"
        self.fidelity = "F0a - empirical lookup"
        self.version = "1.0.0"
        self._curve = TEGEfficiencyCurve(params_path)

    def predict(self, inputs: dict) -> dict:
        T_hot = float(inputs.get("T_hot", self._p["rated_point"]["T_hot"]["value"]))
        T_cold = float(inputs.get("T_cold", self._curve.T_cold_ref))
        eff = self._curve.efficiency(T_hot)
        return {
            "efficiency": eff,
            "delta_T": T_hot - T_cold,
            "unit": "fraction",
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"T_hot": "degC", "T_cold": "degC"},
            "outputs": {"efficiency": "fraction", "delta_T": "K"},
            "valid_ranges": self._p["valid_ranges"],
            "source": self._p["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print(m.predict({"T_hot": 200.0, "T_cold": 30.0}))
