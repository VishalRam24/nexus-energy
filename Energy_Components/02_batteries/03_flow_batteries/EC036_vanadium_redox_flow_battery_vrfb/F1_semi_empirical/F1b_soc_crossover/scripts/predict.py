"""EC036 — VRFB — F1b SOC+Crossover — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import VRFBF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = VRFBF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        soc = np.asarray(inputs["soc"], dtype=float)
        current = np.asarray(inputs["current"], dtype=float)
        cycle = np.asarray(inputs.get("cycle_number", 0), dtype=float)
        soc = np.clip(soc, VRFBF1b.SOC_MIN, VRFBF1b.SOC_MAX)
        return {
            "terminal_voltage": self._model.stack_voltage(soc, current),
            "power": self._model.power_w(soc, current),
            "crossover_current_A": self._model.crossover_current(soc),
            "capacity_fade_pct": self._model.capacity_fade_pct(soc, cycle),
            "coulombic_efficiency": self._model.coulombic_efficiency(soc, current),
        }

    def get_info(self) -> dict:
        return {
            "name": "Vanadium Redox Flow Battery (VRFB)",
            "ec_id": "EC036",
            "fidelity": "F1b",
            "model": "SOC + Crossover Model",
            "description": "Nernst+Ohmic + vanadium crossover through membrane",
            "inputs": {
                "soc": {"unit": "dimensionless", "range": [0.01, 0.99]},
                "current": {"unit": "A", "range": [-100, 100]},
                "cycle_number": {"unit": "cycles", "range": [0, 10000], "default": 0},
            },
            "outputs": {
                "terminal_voltage": {"unit": "V"},
                "power": {"unit": "W"},
                "crossover_current_A": {"unit": "A"},
                "capacity_fade_pct": {"unit": "%"},
                "coulombic_efficiency": {"unit": "dimensionless"},
            },
            "source": "Blanc & Rufer (2010); Skyllas-Kazacos et al. (2011)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"soc": 0.5, "current": 50.0, "cycle_number": 100})
    for k, v in r.items():
        print(f"  {k}: {float(v):.6f}")
