"""
EC035 -- NaS Battery -- F1b SOC-Thermal -- Standardized Predict Interface

HIGH-TEMPERATURE CELL: valid temperature range 573.15-623.15 K (300-350 degC).
Outside this range, 'functional' output is False and voltage/power are 0.

Usage:
    model = ComponentModel()
    result = model.predict({"soc": 0.5, "current": 50.0, "temperature": 593.15})
"""

import json
import numpy as np
from pathlib import Path
from model import NaSBatteryF1b


class ComponentModel:
    """Standardized interface for EC035 NaS Battery -- F1b SOC-thermal model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = NaSBatteryF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "soc": float or array (0-1),
                "current": float or array in A (positive=discharge),
                "temperature": float or array in K (MUST be in 573.15-623.15 K)
            }
        Returns: {
            "terminal_voltage": V (0 if outside 300-350C window),
            "power": W,
            "heat_generation": W,
            "effective_capacity": Ah,
            "internal_resistance": Ohm,
            "ocv": V,
            "dsoc_dt": 1/s,
            "functional": bool (True if T in [300, 350] degC)
        }
        """
        soc = np.asarray(inputs["soc"], dtype=float)
        current = np.asarray(inputs["current"], dtype=float)
        temperature = np.asarray(inputs.get("temperature", 593.15), dtype=float)

        return {
            "terminal_voltage": self._model.terminal_voltage(soc, current, temperature),
            "power": self._model.power(soc, current, temperature),
            "heat_generation": self._model.heat_generation(soc, current, temperature),
            "effective_capacity": self._model.effective_capacity(temperature),
            "internal_resistance": self._model.internal_resistance(temperature),
            "ocv": self._model.ocv(soc),
            "dsoc_dt": self._model.soc_derivative(current, temperature),
            "functional": self._model.is_functional(temperature),
        }

    def get_info(self) -> dict:
        return {
            "name": "Sodium-Sulfur (NaS) Battery",
            "ec_id": "EC035",
            "fidelity": "F1b",
            "description": "High-temperature SOC-thermal model; operational ONLY at 300-350 degC (573-623 K); beta-alumina electrolyte",
            "inputs": {
                "soc": {"unit": "dimensionless", "range": [0.0, 1.0]},
                "current": {"unit": "A", "range": [-200.0, 200.0],
                            "note": "positive=discharge, negative=charge"},
                "temperature": {"unit": "K", "range": [573.15, 623.15],
                                "note": "300 to 350 degC ONLY; cell non-functional outside this range"},
            },
            "outputs": {
                "terminal_voltage": {"unit": "V", "note": "0 if outside operating T window"},
                "power": {"unit": "W"},
                "heat_generation": {"unit": "W"},
                "effective_capacity": {"unit": "Ah"},
                "internal_resistance": {"unit": "Ohm"},
                "ocv": {"unit": "V"},
                "dsoc_dt": {"unit": "1/s"},
                "functional": {"unit": "bool", "note": "True if T in [300, 350] degC"},
            },
            "source": "Wen et al. (2008); Sudworth & Tilley (1985); NGK NAS Brochure",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    result = model.predict({"soc": 0.5, "current": 50.0, "temperature": 593.15})
    print(f"\nAt SOC=0.5, I=50A, T=593.15K (320 degC):")
    for k, v in result.items():
        if hasattr(v, '__len__'):
            print(f"  {k}: {v}")
        else:
            val = bool(v) if k == "functional" else float(v)
            print(f"  {k}: {val}")
