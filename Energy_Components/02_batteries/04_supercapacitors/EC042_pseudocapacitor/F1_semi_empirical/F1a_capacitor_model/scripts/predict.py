"""
EC042 -- Pseudocapacitor -- F1a Capacitor Model -- Standardized Predict Interface

Usage:
    model = ComponentModel()
    result = model.predict({"V0": 2.7, "current": 100.0, "dt": 1.0})
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import PseudocapacitorModel

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    """Standardized interface for EC042 Pseudocapacitor -- F1a capacitor model."""

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = _PARAMS_PATH
        with open(params_path) as f:
            raw = json.load(f)
        self.params = raw.get("default_parameters", raw)
        self._model = PseudocapacitorModel(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "V0": float -- current/initial voltage [V],
                "current": float -- discharge current [A] (positive=discharge, default 0),
                "dt": float -- time step [s] (default 0 = snapshot)
            }
        Returns: {
            "V_terminal": V (with ESR drop),
            "V_new": V (voltage after dt),
            "E_stored_J": J,
            "E_stored_Wh": Wh,
            "P_output": W,
            "SOC": dimensionless (energy-based),
            "efficiency": dimensionless
        }
        """
        V0 = float(inputs.get("V0", self._model.V_max))
        current = float(inputs.get("current", 0.0))
        dt = float(inputs.get("dt", 0.0))
        return self._model.evaluate(V0, I=current, dt=dt)

    def get_info(self) -> dict:
        return {
            "name": "Pseudocapacitor",
            "ec_id": "EC042",
            "fidelity": "F1a",
            "description": "Simple capacitor model: V(t)=V0-I*t/C-I*R_esr, C=500F, R_esr=5mOhm, V_max=2.7V",
            "inputs": {
                "V0": {"unit": "V", "range": [0.0, 2.7], "note": "current voltage"},
                "current": {"unit": "A", "note": "positive=discharge"},
                "dt": {"unit": "s", "note": "time step; 0=instantaneous snapshot"},
            },
            "outputs": {
                "V_terminal": {"unit": "V"},
                "V_new": {"unit": "V"},
                "E_stored_J": {"unit": "J"},
                "E_stored_Wh": {"unit": "Wh"},
                "P_output": {"unit": "W"},
                "SOC": {"unit": "dimensionless", "note": "energy-based: V^2/V_max^2"},
                "efficiency": {"unit": "dimensionless"},
            },
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    result = model.predict({"V0": 2.7, "current": 100.0, "dt": 1.0})
    print("\nAt V0=2.7V, I=100A, dt=1s:")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")
