"""EC043 — Hybrid Supercapacitor — F1a Capacitor Model — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import HybridSupercapacitorF1a


class ComponentModel:
    component_id = "EC043"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = HybridSupercapacitorF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        return self._model.predict(
            Q=np.asarray(inputs["Q"], dtype=float),
            I=np.asarray(inputs["I"], dtype=float),
        )

    def get_info(self) -> dict:
        return {
            "name": "Hybrid Supercapacitor",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": "Simple RC model: V = Q/C + I*R_esr",
            "inputs": {
                "Q": {"unit": "C", "range": [0.0, 760.0]},
                "I": {"unit": "A", "range": [-500.0, 500.0]},
            },
            "outputs": {
                "V_terminal": {"unit": "V"},
                "V_oc": {"unit": "V"},
                "SOC": {"unit": "dimensionless"},
                "P_W": {"unit": "W"},
                "E_Wh": {"unit": "Wh"},
                "V_drop_esr": {"unit": "V"},
            },
            "source": "Conway (1999). Electrochemical Supercapacitors",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"Q": 380.0, "I": 100.0})
    print(f"V_terminal={float(r['V_terminal']):.3f} V, SOC={float(r['SOC']):.2f}, P={float(r['P_W']):.1f} W")
