"""
EC094 -- Evaporative Cooler -- F1a Effectiveness -- Standardized Predict Interface

Usage:
    model = ComponentModel()
    result = model.predict({"T_db": 35.0, "T_wb": 20.0, "m_dot_air": 2.0})
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import EvaporativeCoolerModel

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    """Standardized interface for EC094 Evaporative Cooler -- F1a effectiveness model."""

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = _PARAMS_PATH
        with open(params_path) as f:
            raw = json.load(f)
        self.params = raw.get("default_parameters", raw)
        self._model = EvaporativeCoolerModel(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "T_db": float -- dry-bulb temperature [degC],
                "T_wb": float -- wet-bulb temperature [degC],
                "m_dot_air": float -- air mass flow [kg/s] (default 1.0)
            }
        Returns: {
            "T_out": degC,
            "Q_cool_W": W,
            "COP": dimensionless,
            "delta_T": K,
            "P_fan_W": W
        }
        """
        T_db = float(inputs["T_db"])
        T_wb = float(inputs["T_wb"])
        m_dot = float(inputs.get("m_dot_air", 1.0))
        return self._model.evaluate(T_db, T_wb, m_dot_air=m_dot)

    def get_info(self) -> dict:
        return {
            "name": "Evaporative Cooler",
            "ec_id": "EC094",
            "fidelity": "F1a",
            "description": "Effectiveness model: T_out=T_db-eps*(T_db-T_wb), eps=0.85, P_fan=200W",
            "inputs": {
                "T_db": {"unit": "degC", "note": "dry-bulb temperature"},
                "T_wb": {"unit": "degC", "note": "wet-bulb temperature (must be <= T_db)"},
                "m_dot_air": {"unit": "kg/s", "note": "air mass flow (default 1.0)"},
            },
            "outputs": {
                "T_out": {"unit": "degC"},
                "Q_cool_W": {"unit": "W"},
                "COP": {"unit": "dimensionless"},
                "delta_T": {"unit": "K"},
                "P_fan_W": {"unit": "W"},
            },
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    result = model.predict({"T_db": 35.0, "T_wb": 20.0, "m_dot_air": 2.0})
    print("\nAt T_db=35degC, T_wb=20degC, m_dot=2kg/s:")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")
