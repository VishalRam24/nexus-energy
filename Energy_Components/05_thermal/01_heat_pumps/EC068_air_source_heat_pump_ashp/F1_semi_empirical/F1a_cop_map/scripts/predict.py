"""EC068 — ASHP — F1a COP Map — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import ASHPF1a

class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ASHPF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        Ts = np.asarray(inputs["T_source"], dtype=float)
        Tk = np.asarray(inputs["T_sink"], dtype=float)
        plr = np.asarray(inputs.get("part_load_ratio", 1.0), dtype=float)
        return {
            "cop": self._model.cop(Ts, Tk),
            "heating_capacity_kw": self._model.heating_capacity(Ts, Tk, plr),
            "electrical_input_kw": self._model.electrical_input(Ts, Tk, plr),
        }

    def get_info(self) -> dict:
        return {
            "name": "Air-Source Heat Pump (ASHP)",
            "ec_id": "EC068",
            "fidelity": "F1a",
            "description": "COP = eta_Carnot * T_sink / (T_sink - T_source)",
            "inputs": {
                "T_source": {"unit": "degC", "range": [-20.0, 40.0]},
                "T_sink": {"unit": "degC", "range": [25.0, 65.0]},
                "part_load_ratio": {"unit": "-", "range": [0.0, 1.0], "default": 1.0},
            },
            "outputs": {
                "cop": {"unit": "dimensionless"},
                "heating_capacity_kw": {"unit": "kW"},
                "electrical_input_kw": {"unit": "kW"},
            },
            "source": "Staffell et al. (2012); EN 14511",
            "license": "BSD-3",
        }

if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_source": 7.0, "T_sink": 35.0})
    print(f"At A7/W35: COP={float(r['cop']):.2f}, Q={float(r['heating_capacity_kw']):.1f}kW, W={float(r['electrical_input_kw']):.2f}kW")
