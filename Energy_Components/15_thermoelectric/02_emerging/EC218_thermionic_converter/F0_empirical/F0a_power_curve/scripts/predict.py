"""EC218 Thermionic Converter F0a - standard prediction interface."""
import json
import os

from model import ThermionicPowerCurve


class ComponentModel:
    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        self.params_path = params_path
        with open(params_path) as f:
            self._p = json.load(f)
        self.component_id = "EC218"
        self.component_name = "Thermionic Converter"
        self.fidelity = "F0a - empirical lookup"
        self.version = "1.0.0"
        self._curve = ThermionicPowerCurve(params_path)

    def predict(self, inputs: dict) -> dict:
        Te = float(inputs.get("T_emitter", self._p["rated_point"]["T_emitter"]["value"]))
        pd = self._curve.power_density(Te)
        return {"power_density": pd, "power": self._curve.power(Te), "unit": "W/m^2"}

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"T_emitter": "K"},
            "outputs": {"power_density": "W/m^2", "power": "W"},
            "valid_ranges": self._p["valid_ranges"],
            "source": self._p["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print(m.predict({"T_emitter": 1700.0}))
