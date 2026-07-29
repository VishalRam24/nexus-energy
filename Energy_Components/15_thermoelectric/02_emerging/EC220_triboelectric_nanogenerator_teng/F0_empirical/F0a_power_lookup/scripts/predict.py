"""EC220 TENG F0a - standard prediction interface."""
import json
import os

from model import TENGPowerLookup


class ComponentModel:
    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        self.params_path = params_path
        with open(params_path) as f:
            self._p = json.load(f)
        self.component_id = "EC220"
        self.component_name = "Triboelectric Nanogenerator (TENG)"
        self.fidelity = "F0a - empirical lookup"
        self.version = "1.0.0"
        self._lut = TENGPowerLookup(params_path)

    def predict(self, inputs: dict) -> dict:
        f = float(inputs.get("frequency", self._p["rated_point"]["frequency"]["value"]))
        p_mW = self._lut.power_mW(f)
        return {"power_mW": p_mW, "power_W": p_mW * 1e-3, "unit": "mW"}

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"frequency": "Hz"},
            "outputs": {"power_mW": "mW", "power_W": "W"},
            "valid_ranges": self._p["valid_ranges"],
            "source": self._p["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print(m.predict({"frequency": 3.0}))
