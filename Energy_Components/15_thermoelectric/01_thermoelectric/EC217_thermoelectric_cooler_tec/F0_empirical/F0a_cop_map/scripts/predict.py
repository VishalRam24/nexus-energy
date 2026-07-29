"""EC217 TEC F0a - standard prediction interface."""
import json
import os

from model import TECCopMap


class ComponentModel:
    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        self.params_path = params_path
        with open(params_path) as f:
            self._p = json.load(f)
        self.component_id = "EC217"
        self.component_name = "Thermoelectric Cooler (TEC)"
        self.fidelity = "F0a - empirical lookup"
        self.version = "1.0.0"
        self._map = TECCopMap(params_path)

    def predict(self, inputs: dict) -> dict:
        if "delta_T" in inputs:
            dT = float(inputs["delta_T"])
        else:
            Th = float(inputs.get("Th", self._map.Th_ref))
            Tc = float(inputs.get("Tc", self._p["rated_point"]["Tc"]["value"]))
            dT = Th - Tc
        return {"COP": self._map.cop(dT), "delta_T": dT, "unit": "dimensionless"}

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"delta_T": "K", "Th": "K", "Tc": "K"},
            "outputs": {"COP": "dimensionless", "delta_T": "K"},
            "valid_ranges": self._p["valid_ranges"],
            "source": self._p["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print(m.predict({"delta_T": 30.0}))
