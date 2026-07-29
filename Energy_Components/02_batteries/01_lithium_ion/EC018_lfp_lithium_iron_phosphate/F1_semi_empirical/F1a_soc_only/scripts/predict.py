"""
EC018 — LFP Battery — F1a SOC-Only — Standardized Predict Interface
"""

import json
import numpy as np
from pathlib import Path
from model import LFPBatteryF1a


class ComponentModel:
    """Standardized interface for EC018 LFP Battery — F1a SOC-only voltage model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = LFPBatteryF1a(self.params)

    def predict(self, inputs: dict) -> dict:
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
            "name": "LFP Battery (Lithium Iron Phosphate)",
            "ec_id": "EC018",
            "fidelity": "F1a",
            "description": "Simple SOC-voltage model: V = OCV(SOC) - I*R_internal",
            "inputs": {
                "soc": {"unit": "dimensionless", "range": [0.0, 1.0]},
                "current": {"unit": "A", "range": [-17.5, 17.5]},
            },
            "outputs": {
                "voltage": {"unit": "V"},
                "ocv": {"unit": "V"},
                "power": {"unit": "W"},
                "dsoc_dt": {"unit": "1/s"},
            },
            "source": "Chen et al. (2020); A123 ANR26650M1B datasheet",
            "license": "BSD-3 (model equations)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    result = model.predict({"soc": 0.5, "current": 2.5})
    print(f"\nAt SOC=0.5, I=2.5A (1C discharge):")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")
