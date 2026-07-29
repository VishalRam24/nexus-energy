"""
EC022 — LCO Battery — F1a SOC-Only — Standardized Predict Interface
"""

import json
import numpy as np
from pathlib import Path
from model import LCOBatteryF1a


class ComponentModel:
    """Standardized interface for EC022 LCO Battery — F1a SOC-only voltage model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = LCOBatteryF1a(self.params)

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
            "name": "LCO Battery (Lithium Cobalt Oxide)",
            "ec_id": "EC022",
            "fidelity": "F1a",
            "description": "Simple SOC-voltage model: V = OCV(SOC) - I*R_internal",
            "inputs": {
                "soc": {"unit": "dimensionless", "range": [0.0, 1.0]},
                "current": {"unit": "A", "range": [-10.0, 10.0],
                            "note": "positive=discharge, negative=charge"},
            },
            "outputs": {
                "voltage": {"unit": "V"},
                "ocv": {"unit": "V"},
                "power": {"unit": "W"},
                "dsoc_dt": {"unit": "1/s"},
            },
            "source": "Reimers & Dahn (1992), J. Electrochem. Soc. 139, 2091",
            "license": "BSD-3 (model equations)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    result = model.predict({"soc": 0.5, "current": 1.3})
    print(f"\nAt SOC=0.5, I=1.3A (~0.5C discharge):")
    print(f"  Voltage: {result['voltage']:.4f} V")
    print(f"  OCV:     {result['ocv']:.4f} V")
    print(f"  Power:   {result['power']:.4f} W")
