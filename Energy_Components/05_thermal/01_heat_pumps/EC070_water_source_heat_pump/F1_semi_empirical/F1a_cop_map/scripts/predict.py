"""EC070 — Water-Source Heat Pump — F1a COP Map — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import WaterSourceHPF1a

class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = WaterSourceHPF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        Ts = np.asarray(inputs["T_source"], dtype=float)
        Tk = np.asarray(inputs["T_sink"], dtype=float)
        plr = np.asarray(inputs.get("part_load_ratio", 1.0), dtype=float)
        return {
            "cop": self._model.cop(Ts, Tk),
            "cooling_cop": self._model.cooling_cop(Ts, Tk),
            "heating_capacity_kw": self._model.heating_capacity(Ts, Tk, plr),
            "cooling_capacity_kw": self._model.cooling_capacity(Ts, Tk, plr),
            "electrical_input_kw": self._model.electrical_input(Ts, Tk, plr),
        }

    def get_info(self) -> dict:
        return {
            "name": "Water-Source Heat Pump (WSHP)",
            "ec_id": "EC070",
            "fidelity": "F1a",
            "description": "COP = eta_Carnot * T_sink / (T_sink - T_source); water source (10-25 degC typical)",
            "inputs": {
                "T_source": {"unit": "degC", "range": [5.0, 30.0], "note": "Groundwater/surface water temperature"},
                "T_sink": {"unit": "degC", "range": [25.0, 65.0]},
                "part_load_ratio": {"unit": "-", "range": [0.0, 1.0], "default": 1.0},
            },
            "outputs": {
                "cop": {"unit": "dimensionless", "note": "Heating COP"},
                "cooling_cop": {"unit": "dimensionless"},
                "heating_capacity_kw": {"unit": "kW"},
                "cooling_capacity_kw": {"unit": "kW"},
                "electrical_input_kw": {"unit": "kW"},
            },
            "source": "ASHRAE Handbook — HVAC Systems (2020); EN 14511",
            "license": "BSD-3",
        }

if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"T_source": 15.0, "T_sink": 45.0})
    print(f"\nAt W15/W45: COP={float(r['cop']):.2f}, Q={float(r['heating_capacity_kw']):.1f}kW, W={float(r['electrical_input_kw']):.2f}kW")
