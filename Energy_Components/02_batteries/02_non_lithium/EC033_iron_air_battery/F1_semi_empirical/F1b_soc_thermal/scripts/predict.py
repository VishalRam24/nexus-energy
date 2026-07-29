"""EC033 -- Iron-Air Battery -- F1b SOC-Thermal -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import IronAirBatteryF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = IronAirBatteryF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        soc         = np.asarray(inputs["soc"], dtype=float)
        current     = np.asarray(inputs["current"], dtype=float)
        temperature = np.asarray(inputs.get("temperature", 298.15), dtype=float)
        return {
            "terminal_voltage":    self._model.terminal_voltage(soc, current, temperature),
            "power":               self._model.power(soc, current, temperature),
            "heat_generation":     self._model.heat_generation(soc, current, temperature),
            "effective_capacity":  self._model.effective_capacity(temperature),
            "internal_resistance": self._model.internal_resistance(temperature),
            "ocv":                 self._model.ocv(soc),
            "dsoc_dt":             self._model.soc_derivative(current, temperature),
        }

    def get_info(self) -> dict:
        return {
            "name": "Iron-Air Battery",
            "ec_id": "EC033",
            "fidelity": "F1b",
            "description": "SOC-thermal model: V = OCV(SOC) - I*R(T), Arrhenius R(T), dOCV/dT > 0 (Fe-air)",
            "inputs": {
                "soc":         {"unit": "dimensionless", "range": [0.0, 1.0]},
                "current":     {"unit": "A", "range": [-3.0, 3.0]},
                "temperature": {"unit": "K", "range": [253.15, 333.15]},
            },
            "outputs": {
                "terminal_voltage":    {"unit": "V"},
                "power":               {"unit": "W"},
                "heat_generation":     {"unit": "W"},
                "effective_capacity":  {"unit": "Ah"},
                "internal_resistance": {"unit": "Ohm"},
                "ocv":                 {"unit": "V"},
                "dsoc_dt":             {"unit": "1/s"},
            },
            "source": "Manohar (2012); Trocino (2022); Form Energy (2022)",
        }
