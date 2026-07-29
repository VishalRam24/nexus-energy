"""
EC106 -- SOFC-Based Fuel Cell CHP -- F1a Efficiency Curve -- Standardized Predict Interface

Usage:
    model = ComponentModel()
    result = model.predict({"load_fraction": 1.0})
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import SOFCCHPModel

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    """Standardized interface for EC106 SOFC CHP -- F1a eta-curve model."""

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = _PARAMS_PATH
        with open(params_path) as f:
            raw = json.load(f)
        self.params = raw.get("default_parameters", raw)
        self._model = SOFCCHPModel(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {"load_fraction": float (0-1)}
        Returns: {
            "eta_e": dimensionless,
            "eta_th": dimensionless,
            "P_e_W": W,
            "P_e_kW": kW,
            "Q_th_W": W,
            "Q_in_W": W (fuel heat input),
            "PER": primary energy ratio,
            "load_fraction": dimensionless
        }
        """
        load = float(inputs.get("load_fraction", 1.0))
        return self._model.evaluate(load)

    def get_info(self) -> dict:
        return {
            "name": "SOFC-Based Fuel Cell CHP",
            "ec_id": "EC106",
            "fidelity": "F1a",
            "description": "Eta-curve: eta_e=0.55*f_e(load), eta_th=0.30*f_th(load), P_e_rated=5kW",
            "inputs": {
                "load_fraction": {"unit": "dimensionless", "range": [0.0, 1.0]},
            },
            "outputs": {
                "eta_e": {"unit": "dimensionless"},
                "eta_th": {"unit": "dimensionless"},
                "P_e_W": {"unit": "W"},
                "P_e_kW": {"unit": "kW"},
                "Q_th_W": {"unit": "W"},
                "Q_in_W": {"unit": "W"},
                "PER": {"unit": "dimensionless", "note": "primary energy ratio"},
                "load_fraction": {"unit": "dimensionless"},
            },
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    result = model.predict({"load_fraction": 1.0})
    print("\nAt full load:")
    for k, v in result.items():
        print(f"  {k}: {v}")
