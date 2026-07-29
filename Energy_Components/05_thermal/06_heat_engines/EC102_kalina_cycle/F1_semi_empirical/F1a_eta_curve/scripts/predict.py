"""
EC102 -- Kalina Cycle -- F1a Efficiency Curve -- Standardized Predict Interface

Usage:
    model = ComponentModel()
    result = model.predict({"T_source_K": 423.15})   # 150 degC
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import KalinaCycleModel

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    """Standardized interface for EC102 Kalina Cycle -- F1a eta-curve model."""

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = _PARAMS_PATH
        with open(params_path) as f:
            raw = json.load(f)
        self.params = raw.get("default_parameters", raw)
        self._model = KalinaCycleModel(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "T_source_K": float -- heat source temperature [K],
                "Q_source_W": float -- heat input [W] (optional)
            }
        Returns: {
            "eta": dimensionless,
            "eta_Carnot": dimensionless,
            "P_out_W": W,
            "P_out_kW": kW,
            "Q_source_W": W,
            "Q_rejected_W": W,
            "T_source_K": K
        }
        """
        T_source = float(inputs["T_source_K"])
        Q_source = inputs.get("Q_source_W", None)
        if Q_source is not None:
            Q_source = float(Q_source)
        return self._model.evaluate(T_source, Q_source_W=Q_source)

    def get_info(self) -> dict:
        return {
            "name": "Kalina Cycle",
            "ec_id": "EC102",
            "fidelity": "F1a",
            "description": "Carnot*eta_2nd: eta=0.55*(1-T_sink/T_source), T_source=100-200degC, NH3/H2O",
            "inputs": {
                "T_source_K": {"unit": "K", "range": [373.15, 473.15], "note": "100-200 degC"},
                "Q_source_W": {"unit": "W", "note": "heat input (optional)"},
            },
            "outputs": {
                "eta": {"unit": "dimensionless"},
                "eta_Carnot": {"unit": "dimensionless"},
                "P_out_W": {"unit": "W"},
                "P_out_kW": {"unit": "kW"},
                "Q_source_W": {"unit": "W"},
                "Q_rejected_W": {"unit": "W"},
                "T_source_K": {"unit": "K"},
            },
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    result = model.predict({"T_source_K": 423.15})
    print("\nAt T_source=150 degC (423.15K):")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")
