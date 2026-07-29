"""EC221 MHD Generator F0a - standard prediction interface."""
import json
import os

from model import MHDPowerCurve


class ComponentModel:
    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        self.params_path = params_path
        with open(params_path) as f:
            self._p = json.load(f)
        self.component_id = "EC221"
        self.component_name = "Magnetohydrodynamic (MHD) Generator"
        self.fidelity = "F0a - empirical lookup"
        self.version = "1.0.0"
        self._curve = MHDPowerCurve(params_path)

    def predict(self, inputs: dict) -> dict:
        u = float(inputs.get("u", self._p["rated_point"]["u"]["value"]))
        pd = self._curve.power_density(u)
        return {"power_density": pd, "power_W": self._curve.power(u), "unit": "W/m^3"}

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"u": "m/s"},
            "outputs": {"power_density": "W/m^3", "power_W": "W"},
            "valid_ranges": self._p["valid_ranges"],
            "source": self._p["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print(m.predict({"u": 800.0}))
