"""
EC021 — LTO Battery — F1a SOC-Only — Standardized Predict Interface

Usage:
    model = ComponentModel()
    result = model.predict({"soc": 0.5, "current": 1.5})
"""

import json
import numpy as np
from pathlib import Path
from model import LTOBatteryF1a


class ComponentModel:
    """Standardized interface for EC021 LTO Battery — F1a SOC-only voltage model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = LTOBatteryF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "soc": float or array (0-1),
                "current": float or array in A (positive=discharge)
            }
        Returns:
            {
                "voltage": terminal voltage in V,
                "ocv": open-circuit voltage in V,
                "power": electrical power in W,
                "dsoc_dt": SOC rate of change in 1/s
            }
        """
        soc = np.asarray(inputs["soc"], dtype=float)
        current = np.asarray(inputs["current"], dtype=float)

        return {
            "voltage": self._model.terminal_voltage(soc, current),
            "ocv": self._model.ocv(soc),
            "power": self._model.power(soc, current),
            "dsoc_dt": self._model.soc_derivative(soc, current),
        }

    def get_info(self) -> dict:
        return {
            "name": "LTO Battery (Lithium Titanate Oxide)",
            "ec_id": "EC021",
            "fidelity": "F1a",
            "description": "Simple SOC-voltage model: V = OCV(SOC) - I*R_internal",
            "inputs": {
                "soc": {"unit": "dimensionless", "range": [0.0, 1.0]},
                "current": {"unit": "A", "range": [-30.0, 30.0],
                            "note": "positive=discharge, negative=charge"},
            },
            "outputs": {
                "voltage": {"unit": "V"},
                "ocv": {"unit": "V"},
                "power": {"unit": "W"},
                "dsoc_dt": {"unit": "1/s"},
            },
            "source": "Takami et al. (2011), J. Power Sources 196, 6989",
            "license": "BSD-3 (model equations)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    result = model.predict({"soc": 0.5, "current": 1.5})
    print(f"\nAt SOC=0.5, I=1.5A (~0.5C discharge):")
    print(f"  Voltage: {result['voltage']:.4f} V")
    print(f"  OCV:     {result['ocv']:.4f} V")
    print(f"  Power:   {result['power']:.4f} W")
