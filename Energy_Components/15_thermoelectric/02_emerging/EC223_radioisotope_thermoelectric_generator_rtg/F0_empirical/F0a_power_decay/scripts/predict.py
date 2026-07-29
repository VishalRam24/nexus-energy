"""EC223 RTG F0a - standard prediction interface."""
import json
import os

from model import RTGPowerDecay


class ComponentModel:
    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        self.params_path = params_path
        with open(params_path) as f:
            self._p = json.load(f)
        self.component_id = "EC223"
        self.component_name = "Radioisotope Thermoelectric Generator (RTG)"
        self.fidelity = "F0a - empirical lookup"
        self.version = "1.0.0"
        self._lut = RTGPowerDecay(params_path)

    def predict(self, inputs: dict) -> dict:
        t = float(inputs.get("t_years", self._p["rated_point"]["t"]["value"]))
        return {"power_W": self._lut.power_W(t), "unit": "W"}

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"t_years": "years"},
            "outputs": {"power_W": "W"},
            "valid_ranges": self._p["valid_ranges"],
            "source": self._p["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print(m.predict({"t_years": 45.0}))
