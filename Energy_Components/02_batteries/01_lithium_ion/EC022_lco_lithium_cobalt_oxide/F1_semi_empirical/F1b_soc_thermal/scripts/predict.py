"""
EC022 -- LCO Battery -- F1b SOC-Thermal -- Standardized Predict Interface

Usage:
    model = ComponentModel()
    result = model.predict({"soc": 0.5, "current": 2.6, "temperature": 298.15})
"""

import json
import numpy as np
from pathlib import Path
from model import LCOBatteryF1b


class ComponentModel:
    """Standardized interface for EC022 LCO Battery -- F1b SOC-thermal model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = LCOBatteryF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "soc": float or array (0-1),
                "current": float or array in A (positive=discharge),
                "temperature": float or array in K (default 298.15)
            }
        Returns:
            {
                "terminal_voltage": V,
                "power": W,
                "heat_generation": W,
                "effective_capacity": Ah,
                "internal_resistance": Ohm,
                "ocv": V,
                "dsoc_dt": 1/s
            }
        """
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
            "name": "LCO Battery (Lithium Cobalt Oxide)",
            "ec_id": "EC022",
            "fidelity": "F1b",
            "description": "SOC-thermal voltage model with Arrhenius R(T); LCO consumer-electronics cell, 0-45 degC",
            "inputs": {
                "soc": {"unit": "dimensionless", "range": [0.0, 1.0]},
                "current": {"unit": "A", "range": [-10.0, 10.0],
                            "note": "positive=discharge, negative=charge"},
                "temperature": {"unit": "K", "range": [273.15, 318.15],
                                "note": "0 to 45 degC"},
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
            "source": "Reimers & Dahn (1992); Jalkanen et al. (2015); Thomas & Newman (2003)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    result = model.predict({"soc": 0.5, "current": 2.6, "temperature": 298.15})
    print(f"\nAt SOC=0.5, I=2.6A (1C), T=298.15K:")
    for k, v in result.items():
        val = float(v) if np.ndim(v) == 0 else v
        print(f"  {k}: {val:.4f}")
