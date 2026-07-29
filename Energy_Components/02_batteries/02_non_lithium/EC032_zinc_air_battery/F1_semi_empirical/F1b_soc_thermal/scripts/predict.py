"""EC032 -- Zinc-Air Battery -- F1b SOC-Thermal -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import ZincAirBatteryF1b


class ComponentModel:
    """Standardized interface for EC032 Zinc-Air Battery -- F1b SOC-thermal model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ZincAirBatteryF1b(self.params)

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
            "name": "Zinc-Air Battery",
            "ec_id": "EC032",
            "fidelity": "F1b",
            "description": "SOC-thermal model: V = OCV(SOC) - I*R(T), Arrhenius R(T), dOCV/dT < 0",
            "inputs": {
                "soc":         {"unit": "dimensionless", "range": [0.0, 1.0]},
                "current":     {"unit": "A", "range": [-5.0, 5.0], "note": "positive=discharge"},
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
            "source": "Fu et al. (2010); Lee et al. (2011); Parker et al. (2017)",
        }


if __name__ == "__main__":
    import json
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"soc": 0.5, "current": 1.0, "temperature": 298.15})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
