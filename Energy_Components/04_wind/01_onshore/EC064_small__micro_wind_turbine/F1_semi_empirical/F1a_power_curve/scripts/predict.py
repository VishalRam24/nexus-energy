"""EC064 — Small Wind Turbine — F1a Power Curve — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import SmallWindTurbineF1a


class ComponentModel:
    component_id = "EC064"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SmallWindTurbineF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        return self._model.predict(
            v=np.asarray(inputs["v"], dtype=float),
            rho=np.asarray(inputs.get("rho", self.params["unit"]["rho"]["value"]), dtype=float),
        )

    def get_info(self) -> dict:
        return {
            "name": "Small/Micro Wind Turbine",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": "Power curve: P = 0.5*rho*Cp*A*v^3, with cut-in/rated/cut-out",
            "inputs": {
                "v": {"unit": "m/s", "range": [0.0, 50.0]},
                "rho": {"unit": "kg/m3", "range": [0.9, 1.4]},
            },
            "outputs": {
                "P_kW": {"unit": "kW"},
                "Cp_actual": {"unit": "dimensionless"},
                "P_aero_kW": {"unit": "kW"},
            },
            "source": "Manwell et al. (2009). Wind Energy Explained; IEC 61400-2",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"v": 10.0})
    print(f"P={float(r['P_kW']):.2f} kW at v=10 m/s")
