"""EC056 — Linear Fresnel CSP — F1a Optical Efficiency — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import LinearFresnelF1a


class ComponentModel:
    component_id = "EC056"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = LinearFresnelF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        return self._model.predict(
            DNI=np.asarray(inputs["DNI"], dtype=float),
            theta=np.asarray(inputs.get("theta", 0.0), dtype=float),
            eta_PB=float(inputs.get("eta_PB", 0.33)),
        )

    def get_info(self) -> dict:
        return {
            "name": "Linear Fresnel CSP",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": "Q = eta_opt * eta_th * DNI * A_aperture * cos(theta)",
            "inputs": {
                "DNI": {"unit": "W/m2", "range": [0.0, 1100.0]},
                "theta": {"unit": "deg", "range": [0.0, 60.0]},
                "eta_PB": {"unit": "dimensionless", "range": [0.25, 0.45]},
            },
            "outputs": {
                "Q_MW": {"unit": "MW"},
                "P_MW": {"unit": "MW"},
                "eta_optical": {"unit": "dimensionless"},
                "eta_system": {"unit": "dimensionless"},
            },
            "source": "Morin et al. (2012). Solar Energy",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"DNI": 800.0, "theta": 10.0})
    print(f"Q={float(r['Q_MW']):.2f} MW, P={float(r['P_MW']):.2f} MW")
