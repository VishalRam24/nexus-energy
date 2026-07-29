"""
EC018 -- LFP Battery -- F1b SOC-Thermal -- Standardized Predict Interface

Usage:
    model = ComponentModel()
    result = model.predict({"soc": 0.5, "current": 2.0, "temperature": 298.15})
"""

import json
import numpy as np
from pathlib import Path
from model import LFPBatteryF1b


class ComponentModel:
    """Standardized interface for EC018 LFP Battery -- F1b SOC-thermal model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = LFPBatteryF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        soc = np.asarray(inputs["soc"], dtype=float)
        current = np.asarray(inputs["current"], dtype=float)
        temperature = np.asarray(inputs.get("temperature", 298.15), dtype=float)

        return {
            "terminal_voltage": self._model.terminal_voltage(soc, current, temperature),
            "power": self._model.power(soc, current, temperature),
            "heat_generation": self._model.heat_generation(soc, current, temperature),
            "effective_capacity": self._model.effective_capacity(temperature),
            "internal_resistance": self._model.internal_resistance(temperature),
            "ocv": self._model.ocv(soc),
            "dsoc_dt": self._model.soc_derivative(current, temperature),
        }

    def get_info(self) -> dict:
        return {
            "name": "LFP Battery (Lithium Iron Phosphate)",
            "ec_id": "EC018",
            "fidelity": "F1b",
            "description": "SOC-thermal voltage model with Arrhenius R(T) and capacity correction",
            "inputs": {
                "soc": {"unit": "dimensionless", "range": [0.0, 1.0]},
                "current": {"unit": "A", "range": [-17.5, 17.5],
                            "note": "positive=discharge, negative=charge"},
                "temperature": {"unit": "K", "range": [253.15, 333.15],
                                "note": "-20 to 60 degC"},
            },
            "outputs": {
                "terminal_voltage": {"unit": "V"},
                "power": {"unit": "W"},
                "heat_generation": {"unit": "W"},
                "effective_capacity": {"unit": "Ah"},
                "internal_resistance": {"unit": "Ohm"},
                "ocv": {"unit": "V"},
                "dsoc_dt": {"unit": "1/s"},
            },
            "source": "Chen et al. (2020); Ecker et al. (2015); Thomas et al. (2008)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    result = model.predict({"soc": 0.5, "current": 2.0, "temperature": 298.15})
    print(f"\nAt SOC=0.5, I=2.0A, T=298.15K:")
    for k, v in result.items():
        val = float(v) if np.ndim(v) == 0 else v
        print(f"  {k}: {val:.4f}")
