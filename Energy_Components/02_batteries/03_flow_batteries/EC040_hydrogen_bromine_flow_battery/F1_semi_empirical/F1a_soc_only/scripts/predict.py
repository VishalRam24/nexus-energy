"""
EC040 -- Hydrogen-Bromine Flow Battery -- F1a SOC-only -- Standardized Predict Interface

Usage:
    model = ComponentModel()
    result = model.predict({"soc": 0.5, "current": 30.0})
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import H2BrFlowBatteryModel

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    """Standardized interface for EC040 H2-Br Flow Battery -- F1a SOC-only model."""

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = _PARAMS_PATH
        with open(params_path) as f:
            raw = json.load(f)
        self.params = raw.get("default_parameters", raw)
        self._model = H2BrFlowBatteryModel(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "soc": float (0-1),
                "current": float in A (positive=discharge, default 0),
                "dt": float time step in s (default 0)
            }
        Returns: {
            "V_cell_ocv": V,
            "V_stack_ocv": V,
            "V_stack_terminal": V,
            "P_stack": W,
            "SOC_new": dimensionless,
            "efficiency": dimensionless
        }
        """
        soc = float(inputs["soc"])
        current = float(inputs.get("current", 0.0))
        dt = float(inputs.get("dt", 0.0))
        return self._model.evaluate(soc, I=current, dt=dt)

    def get_info(self) -> dict:
        return {
            "name": "Hydrogen-Bromine Flow Battery",
            "ec_id": "EC040",
            "fidelity": "F1a",
            "description": "Nernst OCV model; 30-cell stack, n=2, H2/Br2, fixed T=298K",
            "inputs": {
                "soc": {"unit": "dimensionless", "range": [0.01, 0.99]},
                "current": {"unit": "A", "range": [-200.0, 200.0], "note": "positive=discharge"},
                "dt": {"unit": "s", "note": "time step for SOC update; 0=no update"},
            },
            "outputs": {
                "V_cell_ocv": {"unit": "V"},
                "V_stack_ocv": {"unit": "V"},
                "V_stack_terminal": {"unit": "V"},
                "P_stack": {"unit": "W"},
                "SOC_new": {"unit": "dimensionless"},
                "efficiency": {"unit": "dimensionless"},
            },
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    result = model.predict({"soc": 0.5, "current": 30.0})
    print("\nAt SOC=0.5, I=30A:")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")
