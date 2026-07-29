"""EC067 — AWE — F1a Power Curve — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import AWEF1a


class ComponentModel:
    component_id = "EC067"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AWEF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        return self._model.predict(
            v=np.asarray(inputs["v"], dtype=float),
            rho=np.asarray(inputs.get("rho", self.params["unit"]["rho"]["value"]), dtype=float),
        )

    def get_info(self) -> dict:
        return {
            "name": "Airborne Wind Energy (AWE)",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": "Crosswind kite power curve: P = 0.5*rho*Cp_eff*A*v^3",
            "inputs": {
                "v": {"unit": "m/s", "range": [0.0, 50.0]},
                "rho": {"unit": "kg/m3", "range": [0.9, 1.4]},
            },
            "outputs": {
                "P_kW": {"unit": "kW"},
                "P_rated_kW": {"unit": "kW"},
                "capacity_factor": {"unit": "dimensionless"},
            },
            "source": "Loyd (1980). J. Energy; Diehl (2013). Springer.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"v": 12.0})
    print(f"P={float(r['P_kW']):.2f} kW at v=12 m/s")
