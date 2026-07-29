"""EC068 — ASHP — F1b Part-Load — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import ASHPF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ASHPF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        Ts = np.asarray(inputs["T_source"], dtype=float)
        Tk = np.asarray(inputs["T_sink"], dtype=float)
        plr = np.asarray(inputs.get("part_load_ratio", 1.0), dtype=float)
        return {
            "cop": self._model.cop(Ts, Tk, plr),
            "heating_capacity_kw": self._model.heating_capacity(Ts, Tk, plr),
            "electrical_input_kw": self._model.electrical_input(Ts, Tk, plr),
            "cop_degradation_factor": self._model.cop_degradation_factor(plr),
        }

    def get_info(self) -> dict:
        return {
            "name": "Air-Source Heat Pump (ASHP)",
            "ec_id": "EC068",
            "fidelity": "F1b",
            "description": "Part-load COP with PLF = 1 - C_d*(1-PLR) per EN 14825, plus cycling losses",
            "inputs": {
                "T_source": {"unit": "degC", "range": [-20.0, 40.0]},
                "T_sink": {"unit": "degC", "range": [25.0, 65.0]},
                "part_load_ratio": {"unit": "-", "range": [0.0, 1.0], "default": 1.0},
            },
            "outputs": {
                "cop": {"unit": "dimensionless"},
                "heating_capacity_kw": {"unit": "kW"},
                "electrical_input_kw": {"unit": "kW"},
                "cop_degradation_factor": {"unit": "dimensionless"},
            },
            "source": "Staffell et al. (2012); EN 14825:2016; AHRI 210/240",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for plr in [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]:
        r = model.predict({"T_source": 7.0, "T_sink": 35.0, "part_load_ratio": plr})
        print(f"PLR={plr:.2f}: COP={float(r['cop']):.2f}, "
              f"Q={float(r['heating_capacity_kw']):.1f}kW, "
              f"W={float(r['electrical_input_kw']):.2f}kW, "
              f"degradation={float(r['cop_degradation_factor']):.3f}")
