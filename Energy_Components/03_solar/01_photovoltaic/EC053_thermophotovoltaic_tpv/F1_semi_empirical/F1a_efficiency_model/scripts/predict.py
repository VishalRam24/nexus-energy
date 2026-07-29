"""EC053 — TPV — F1a Efficiency Model — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import TPVF1a


class ComponentModel:
    component_id = "EC053"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = TPVF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        return self._model.predict(
            T_emitter=np.asarray(inputs.get("T_emitter", self.params["unit"]["T_emitter"]["value"]), dtype=float),
            F_view=np.asarray(inputs.get("F_view", self.params["unit"]["F_view"]["value"]), dtype=float),
        )

    def get_info(self) -> dict:
        return {
            "name": "Thermophotovoltaic (TPV)",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": "P = eta * epsilon * sigma * T_emitter^4 * A_cell * F_view",
            "inputs": {
                "T_emitter": {"unit": "K", "range": [800.0, 2000.0]},
                "F_view": {"unit": "dimensionless", "range": [0.1, 1.0]},
            },
            "outputs": {
                "P_W": {"unit": "W"},
                "P_incident_W": {"unit": "W"},
                "irradiance_Wm2": {"unit": "W/m2"},
                "eta_sys": {"unit": "dimensionless"},
            },
            "source": "Coutts (1999). Renew. Sust. Energy Rev.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"T_emitter": 1500.0})
    print(f"P={float(r['P_W']):.4f} W, irr={float(r['irradiance_Wm2']):.0f} W/m²")
