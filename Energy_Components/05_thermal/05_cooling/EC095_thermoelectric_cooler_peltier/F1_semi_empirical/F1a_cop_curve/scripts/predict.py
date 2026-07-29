"""EC095 — Thermoelectric Cooler (Peltier) — F1a COP Curve — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import PeltierTECF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PeltierTECF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        I  = np.asarray(inputs["current"], dtype=float)
        Tc = np.asarray(inputs["T_cold"],  dtype=float)
        Th = np.asarray(inputs["T_hot"],   dtype=float)
        return {
            "cooling_power_w":      self._model.cooling_power(I, Tc, Th),
            "electrical_input_w":   self._model.electrical_input(I, Tc, Th),
            "heat_rejection_w":     self._model.heat_rejection(I, Tc, Th),
            "cop":                  self._model.cop(I, Tc, Th),
            "i_optimum_a":          self._model.optimum_current(Tc, Th),
        }

    def get_info(self) -> dict:
        return {
            "name": "Thermoelectric Cooler (Peltier, Bi2Te3 stack)",
            "ec_id": "EC095",
            "fidelity": "F1a",
            "description": "Q_c = N*(alpha*I*T_c - 0.5*I^2*R - K*ΔT); W = N*(alpha*I*ΔT + I^2*R); COP_c = Q_c/W",
            "inputs": {
                "current": {"unit": "A",    "range": [0.0, 6.0]},
                "T_cold":  {"unit": "degC", "range": [-20.0, 30.0]},
                "T_hot":   {"unit": "degC", "range": [10.0, 80.0]},
            },
            "outputs": {
                "cooling_power_w":    {"unit": "W"},
                "electrical_input_w": {"unit": "W"},
                "heat_rejection_w":   {"unit": "W"},
                "cop":                {"unit": "dimensionless"},
                "i_optimum_a":        {"unit": "A"},
            },
            "source": "Goldsmid (2010); Rowe Handbook (2006); Riffat & Ma (2003)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"current": 3.0, "T_cold": 5.0, "T_hot": 35.0})
    print(f"At I=3A, T_c=5C, T_h=35C: Q_c={float(r['cooling_power_w']):.1f}W, "
          f"W={float(r['electrical_input_w']):.1f}W, "
          f"COP={float(r['cop']):.2f}, I_opt={float(r['i_optimum_a']):.2f}A")
