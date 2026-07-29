"""EC063 — VAWT — F1a Power Curve — Standardized Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import VAWTF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = VAWTF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        v = np.asarray(inputs["wind_speed"], dtype=float)
        rho = np.asarray(inputs.get("air_density", 1.225), dtype=float)
        return {
            "power_kw": self._model.power(v, rho),
            "capacity_factor": self._model.capacity_factor(v, rho),
            "power_coefficient": self._model.power_coefficient(v, rho),
        }

    def get_info(self) -> dict:
        return {
            "name": "Vertical Axis Wind Turbine (VAWT)",
            "ec_id": "EC063",
            "fidelity": "F1a",
            "description": "Omnidirectional power curve P(v) with air density correction; VAWT Cp ~0.20-0.35",
            "inputs": {
                "wind_speed": {"unit": "m/s", "range": [0.0, 30.0]},
                "air_density": {"unit": "kg/m3", "range": [0.9, 1.4], "default": 1.225},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "capacity_factor": {"unit": "dimensionless"},
                "power_coefficient": {"unit": "dimensionless"},
            },
            "source": "Tjiu et al. (2015) Renew. Energy 75:50-67; Sutherland et al. SAND2012-0304",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"wind_speed": 10.0})
    print(f"\nAt 10 m/s: P={float(r['power_kw']):.2f} kW, CF={float(r['capacity_factor']):.3f}, Cp={float(r['power_coefficient']):.3f}")
