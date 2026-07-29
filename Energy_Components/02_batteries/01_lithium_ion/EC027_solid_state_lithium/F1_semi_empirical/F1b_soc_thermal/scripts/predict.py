"""
EC027 -- Solid-State Lithium Battery -- F1b SOC-Thermal -- Standardized Predict Interface

Usage:
    model = ComponentModel()
    result = model.predict({"soc": 0.5, "current": 4.0, "temperature": 298.15})
"""

import json
import numpy as np
from pathlib import Path
from model import SolidStateLiBatteryF1b


class ComponentModel:
    """Standardized interface for EC027 Solid-State Li Battery -- F1b SOC-thermal model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SolidStateLiBatteryF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "soc": float or array (0-1),
                "current": float or array in A (positive=discharge),
                "temperature": float or array in K (default 298.15)
            }
        Returns: {terminal_voltage, power, heat_generation, effective_capacity,
                  internal_resistance, ocv, dsoc_dt}
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
            "name": "Solid-State Lithium Battery",
            "ec_id": "EC027",
            "fidelity": "F1b",
            "description": "SOC-thermal model; SSE Arrhenius R(T) with E_a=35 kJ/mol (sulfide electrolyte), 0-60 degC",
            "inputs": {
                "soc": {"unit": "dimensionless", "range": [0.0, 1.0]},
                "current": {"unit": "A", "range": [-20.0, 20.0],
                            "note": "positive=discharge, negative=charge"},
                "temperature": {"unit": "K", "range": [273.15, 333.15],
                                "note": "0 to 60 degC"},
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
            "source": "Janek & Zeier (2016); Takada (2013); Thomas & Newman (2003)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    result = model.predict({"soc": 0.5, "current": 4.0, "temperature": 298.15})
    print(f"\nAt SOC=0.5, I=4.0A (1C), T=298.15K:")
    for k, v in result.items():
        val = float(v) if np.ndim(v) == 0 else v
        print(f"  {k}: {val:.4f}")
