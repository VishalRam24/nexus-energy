"""
EC096 -- Magnetic Refrigeration -- F1a COP Model -- Standardized Predict Interface

Usage:
    model = ComponentModel()
    result = model.predict({"W_input_W": 1000.0, "T_cold_K": 268.15, "T_hot_K": 298.15})
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import MagneticRefrigerationModel

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    """Standardized interface for EC096 Magnetic Refrigeration -- F1a COP model."""

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = _PARAMS_PATH
        with open(params_path) as f:
            raw = json.load(f)
        self.params = raw.get("default_parameters", raw)
        self._model = MagneticRefrigerationModel(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "W_input_W": float -- electrical work input [W],
                "T_cold_K": float -- cold side temperature [K] (optional),
                "T_hot_K": float -- hot side temperature [K] (optional)
            }
        Returns: {
            "COP": dimensionless,
            "COP_Carnot": dimensionless,
            "Q_cool_W": W,
            "Q_hot_W": W,
            "T_span_K": K,
            "W_input_W": W
        }
        """
        W = float(inputs.get("W_input_W", 1000.0))
        T_cold = inputs.get("T_cold_K", None)
        T_hot = inputs.get("T_hot_K", None)
        if T_cold is not None:
            T_cold = float(T_cold)
        if T_hot is not None:
            T_hot = float(T_hot)
        return self._model.evaluate(W, T_cold_K=T_cold, T_hot_K=T_hot)

    def get_info(self) -> dict:
        return {
            "name": "Magnetic Refrigeration",
            "ec_id": "EC096",
            "fidelity": "F1a",
            "description": "COP model: COP=eta_2nd*T_cold/(T_hot-T_cold), eta_2nd=0.4, N_stages=6, dT_MCE=3K",
            "inputs": {
                "W_input_W": {"unit": "W", "note": "electrical work input"},
                "T_cold_K": {"unit": "K", "note": "cold side temperature (default 273.15K)"},
                "T_hot_K": {"unit": "K", "note": "hot side temperature (default 298.15K)"},
            },
            "outputs": {
                "COP": {"unit": "dimensionless"},
                "COP_Carnot": {"unit": "dimensionless"},
                "Q_cool_W": {"unit": "W"},
                "Q_hot_W": {"unit": "W"},
                "T_span_K": {"unit": "K"},
                "W_input_W": {"unit": "W"},
            },
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    result = model.predict({"W_input_W": 1000.0})
    print("\nAt W=1000W, T_cold=273.15K, T_hot=298.15K:")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")
