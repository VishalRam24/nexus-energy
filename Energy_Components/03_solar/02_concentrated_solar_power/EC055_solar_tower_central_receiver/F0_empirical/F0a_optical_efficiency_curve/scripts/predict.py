"""Standard ComponentModel interface for EC055 Solar Tower Central Receiver CSP (F0a)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import EfficiencyCurve  # noqa: E402

_HERE = os.path.dirname(__file__)
_PARAMS = os.path.join(_HERE, "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC055"
    component_name = "Solar Tower Central Receiver CSP"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.params = json.load(f)
        r = self.params["rated"]
        self.eta0 = float(r["eta0"]["value"])
        self.irr_key = r["irradiance_key"]["value"]
        self.output_type = r["output_type"]["value"]
        self.curve = EfficiencyCurve(
            eta0=r["eta0"]["value"], a1=r["a1"]["value"], a2=r["a2"]["value"])

    def predict(self, inputs: dict) -> dict:
        """inputs: DNI [W/m2], T_mean [degC], T_ambient [degC] (or delta_T [K])."""
        g = float(inputs.get(self.irr_key, inputs.get("irradiance", 1000.0)))
        if "delta_T" in inputs:
            dt = float(inputs["delta_T"])
        else:
            dt = float(inputs.get("T_mean", 25.0)) - float(inputs.get("T_ambient", 25.0))
        return {
            "power_density": float(self.curve.power_density(g, dt)),
            "efficiency": float(self.curve.efficiency(g, dt)),
            "output_type": self.output_type,
            "power_density_unit": "W/m2",
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {self.irr_key: "W/m2", "T_mean": "degC", "T_ambient": "degC"},
            "outputs": {"power_density": "W/m2", "efficiency": "dimensionless"},
            "eta0": self.eta0,
            "output_type": self.output_type,
            "source": self.params["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print("INFO:", json.dumps(m.get_info(), indent=2))
    print("Peak:", m.predict({m.irr_key: 1000.0, "T_mean": 25.0, "T_ambient": 25.0}))
    print("Hot :", m.predict({m.irr_key: 1000.0, "T_mean": 90.0, "T_ambient": 25.0}))
