"""
EC100 -- Brayton Cycle Gas Turbine -- F1a Efficiency Curve -- Standardized Predict Interface

Usage:
    model = ComponentModel()
    result = model.predict({"load_fraction": 0.8})
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import BraytonCycleModel

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    """Standardized interface for EC100 Brayton Cycle Gas Turbine -- F1a eta-curve model."""

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = _PARAMS_PATH
        with open(params_path) as f:
            raw = json.load(f)
        self.params = raw.get("default_parameters", raw)
        self._model = BraytonCycleModel(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {"load_fraction": float (0-1)}
        Returns: {
            "eta": dimensionless,
            "P_out_W": W,
            "P_out_MW": MW,
            "Q_in_W": W (fuel heat input),
            "Q_exhaust_W": W,
            "load_fraction": dimensionless,
            "f_load": dimensionless (part-load correction)
        }
        """
        load = float(inputs.get("load_fraction", 1.0))
        return self._model.evaluate(load)

    def get_info(self) -> dict:
        return {
            "name": "Brayton Cycle Gas Turbine",
            "ec_id": "EC100",
            "fidelity": "F1a",
            "description": "Eta-curve: eta=eta_rated*(0.2+0.8*load), eta_rated=0.38, P_rated=50MW",
            "inputs": {
                "load_fraction": {"unit": "dimensionless", "range": [0.0, 1.0]},
            },
            "outputs": {
                "eta": {"unit": "dimensionless"},
                "P_out_W": {"unit": "W"},
                "P_out_MW": {"unit": "MW"},
                "Q_in_W": {"unit": "W"},
                "Q_exhaust_W": {"unit": "W"},
                "load_fraction": {"unit": "dimensionless"},
                "f_load": {"unit": "dimensionless"},
            },
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    result = model.predict({"load_fraction": 1.0})
    print("\nAt full load:")
    for k, v in result.items():
        print(f"  {k}: {v}")
