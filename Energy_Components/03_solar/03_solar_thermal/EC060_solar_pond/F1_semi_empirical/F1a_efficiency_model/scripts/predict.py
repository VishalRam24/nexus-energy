"""EC060 — Solar Pond — F1a Efficiency Model — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import SolarPondF1a


class ComponentModel:
    component_id = "EC060"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SolarPondF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        return self._model.predict(
            G=np.asarray(inputs["G"], dtype=float),
            Tm=np.asarray(inputs.get("Tm", 80.0), dtype=float),
            Ta=np.asarray(inputs.get("Ta", 20.0), dtype=float),
        )

    def get_info(self) -> dict:
        return {
            "name": "Solar Pond",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": "Hottel-Whillier: Q = A*[eta0*G - a1*dT - a2*dT^2]",
            "inputs": {
                "G": {"unit": "W/m2", "range": [0.0, 1100.0]},
                "Tm": {"unit": "degC", "range": [30.0, 100.0]},
                "Ta": {"unit": "degC", "range": [-10.0, 40.0]},
            },
            "outputs": {
                "Q_kW": {"unit": "kW"},
                "eta": {"unit": "dimensionless"},
                "dT": {"unit": "degC"},
            },
            "source": "Duffie & Beckman (2013). Solar Engineering of Thermal Processes",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"G": 700.0, "Tm": 80.0, "Ta": 20.0})
    print(f"Q={float(r['Q_kW']):.1f} kW, eta={float(r['eta']):.4f}")
