"""EC057 — Stirling Dish CSP — F1a Optical Efficiency — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import StirlingDishF1a


class ComponentModel:
    component_id = "EC057"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = StirlingDishF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        return self._model.predict(
            DNI=np.asarray(inputs["DNI"], dtype=float),
            theta=np.asarray(inputs.get("theta", 0.0), dtype=float),
        )

    def get_info(self) -> dict:
        return {
            "name": "Stirling Dish CSP",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": "P = eta_opt * eta_engine * DNI * A_dish * cos(theta)",
            "inputs": {
                "DNI": {"unit": "W/m2", "range": [0.0, 1100.0]},
                "theta": {"unit": "deg", "range": [0.0, 45.0]},
            },
            "outputs": {
                "P_kW": {"unit": "kW"},
                "Q_focal_kW": {"unit": "kW"},
                "eta_optical": {"unit": "dimensionless"},
                "eta_system": {"unit": "dimensionless"},
            },
            "source": "Mancini et al. (2003). ASME J. Sol. Energy Eng.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"DNI": 900.0, "theta": 0.0})
    print(f"P={float(r['P_kW']):.2f} kW, Q_focal={float(r['Q_focal_kW']):.2f} kW")
