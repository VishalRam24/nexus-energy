"""EC049 — Multi-Junction CPV — F1a Efficiency Model — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import MultiJunctionCPVF1a


class ComponentModel:
    component_id = "EC049"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MultiJunctionCPVF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        return self._model.predict(
            DNI=np.asarray(inputs["DNI"], dtype=float),
            T_cell=np.asarray(inputs.get("T_cell", 25.0), dtype=float),
            theta_incidence=np.asarray(inputs.get("theta_incidence", 0.0), dtype=float),
        )

    def get_info(self) -> dict:
        return {
            "name": "Multi-Junction Concentrator PV",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": "P = eta * DNI * C_ratio * A_cell * cos(theta)",
            "inputs": {
                "DNI": {"unit": "W/m2", "range": [0.0, 1100.0]},
                "T_cell": {"unit": "degC", "range": [-20.0, 100.0]},
                "theta_incidence": {"unit": "deg", "range": [0.0, 60.0]},
            },
            "outputs": {
                "P_W": {"unit": "W"},
                "eta_eff": {"unit": "dimensionless"},
                "irr_concentrated": {"unit": "W/m2"},
                "P_max_ref": {"unit": "W"},
            },
            "source": "Luque & Hegedus (2011). Handbook of Photovoltaic Science",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"DNI": 900.0, "T_cell": 40.0, "theta_incidence": 0.0})
    print(f"P={float(r['P_W']):.3f} W, eta={float(r['eta_eff']):.3f}")
