"""EC076 — Regenerative HX — F1a e-NTU — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import RegenerativeHXF1a


class ComponentModel:
    component_id = "EC076"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = RegenerativeHXF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        u = self.params["unit"]
        return self._model.predict(
            T_h_in=np.asarray(inputs["T_h_in"], dtype=float),
            T_c_in=np.asarray(inputs["T_c_in"], dtype=float),
            C_h=np.asarray(inputs.get("C_h", u["C_h"]["value"]), dtype=float),
            C_c=np.asarray(inputs.get("C_c", u["C_c"]["value"]), dtype=float),
        )

    def get_info(self) -> dict:
        return {
            "name": "Regenerative Heat Exchanger",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": "Counterflow e-NTU: Q = eps*C_min*(T_h_in - T_c_in)",
            "inputs": {
                "T_h_in": {"unit": "degC", "range": [50.0, 500.0]},
                "T_c_in": {"unit": "degC", "range": [-20.0, 100.0]},
                "C_h": {"unit": "W/K", "range": [500.0, 10000.0]},
                "C_c": {"unit": "W/K", "range": [500.0, 10000.0]},
            },
            "outputs": {
                "Q_kW": {"unit": "kW"},
                "T_h_out": {"unit": "degC"},
                "T_c_out": {"unit": "degC"},
                "effectiveness": {"unit": "dimensionless"},
                "NTU": {"unit": "dimensionless"},
            },
            "source": "Incropera & DeWitt (2006); Kays & London (1984)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"T_h_in": 200.0, "T_c_in": 20.0})
    print(f"Q={float(r['Q_kW']):.1f} kW, eps={float(r['effectiveness']):.3f}, NTU={float(r['NTU']):.2f}")
