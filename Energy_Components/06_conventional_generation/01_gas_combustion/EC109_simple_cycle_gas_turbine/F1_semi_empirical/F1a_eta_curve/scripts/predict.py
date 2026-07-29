"""EC109 — Simple Cycle Gas Turbine — F1a Efficiency Curve — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import SimpleCycleGasTurbineF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SimpleCycleGasTurbineF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        PLR   = np.asarray(inputs["part_load_ratio"], dtype=float)
        T_amb = np.asarray(inputs["ambient_temp_c"],  dtype=float)
        return {
            "power_mw":         self._model.power_mw(PLR),
            "efficiency":       self._model.efficiency(PLR, T_amb),
            "fuel_rate_kgs":    self._model.fuel_rate_kgs(PLR, T_amb),
            "heat_rate_kjkwh":  self._model.heat_rate_kjkwh(PLR, T_amb),
        }

    def get_info(self) -> dict:
        return {
            "name": "Simple Cycle Gas Turbine",
            "ec_id": "EC109",
            "fidelity": "F1a",
            "description": "eta(PLR, T_amb) = eta_rated * f_PLR * f_amb; GE LM6000-class",
            "inputs": {
                "part_load_ratio":  {"unit": "dimensionless", "range": [0.3, 1.0]},
                "ambient_temp_c":   {"unit": "degC",          "range": [-20.0, 50.0]},
            },
            "outputs": {
                "power_mw":        {"unit": "MW"},
                "efficiency":      {"unit": "dimensionless"},
                "fuel_rate_kgs":   {"unit": "kg/s"},
                "heat_rate_kjkwh": {"unit": "kJ/kWh"},
            },
            "source": "Walsh & Fletcher (2004), Gas Turbine Performance, 2nd ed.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp_c": 15.0})
    print(f"At ISO full load: P={float(r['power_mw']):.1f} MW, eta={float(r['efficiency']):.3f}, "
          f"fuel={float(r['fuel_rate_kgs']):.2f} kg/s, HR={float(r['heat_rate_kjkwh']):.0f} kJ/kWh")
