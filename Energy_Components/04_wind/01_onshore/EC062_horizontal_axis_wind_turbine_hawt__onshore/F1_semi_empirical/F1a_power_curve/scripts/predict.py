"""EC062 — HAWT Onshore — F1a Power Curve — Standardized Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import HAWTOnshoreF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = HAWTOnshoreF1a(self.params)

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
            "name": "Horizontal Axis Wind Turbine (HAWT) — Onshore",
            "ec_id": "EC062",
            "fidelity": "F1a",
            "description": "Power curve P(v) with air density correction",
            "inputs": {
                "wind_speed": {"unit": "m/s", "range": [0.0, 30.0]},
                "air_density": {"unit": "kg/m3", "range": [0.9, 1.4], "default": 1.225},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "capacity_factor": {"unit": "dimensionless"},
                "power_coefficient": {"unit": "dimensionless"},
            },
            "source": "IEC 61400-12-1; Vestas V90-2.0MW datasheet; windpowerlib",
            "license": "MIT (windpowerlib), BSD-3 (model equations)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"wind_speed": 10.0})
    print(f"\nAt 10 m/s: P={float(r['power_kw']):.0f} kW, CF={float(r['capacity_factor']):.3f}, Cp={float(r['power_coefficient']):.3f}")
