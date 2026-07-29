"""EC069 — GSHP — F1b Part-Load — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import GSHPF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = GSHPF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        plr = np.asarray(inputs.get("part_load_ratio", 1.0), dtype=float)
        T_sink = np.asarray(inputs["T_sink"], dtype=float)

        # Accept either T_ground directly or month (for seasonal model)
        if "T_ground" in inputs:
            T_src = np.asarray(inputs["T_ground"], dtype=float)
        elif "month" in inputs:
            T_src = self._model.ground_temperature(inputs["month"])
        else:
            T_src = np.asarray(inputs.get("T_source", 10.0), dtype=float)

        return {
            "cop": self._model.cop(T_src, T_sink, plr),
            "heating_capacity_kw": self._model.heating_capacity(T_src, T_sink, plr),
            "electrical_input_kw": self._model.electrical_input(T_src, T_sink, plr),
            "T_ground": T_src,
        }

    def get_info(self) -> dict:
        return {
            "name": "Ground-Source Heat Pump (GSHP)",
            "ec_id": "EC069",
            "fidelity": "F1b",
            "description": "Part-load PLF (EN 14825) + seasonal ground temperature T(month) = T_mean - A*cos(2pi*(month-month_min)/12)",
            "inputs": {
                "T_ground": {"unit": "degC", "range": [0.0, 20.0], "note": "Direct ground temp, OR use month"},
                "month": {"unit": "month", "range": [1, 12], "note": "Alternative to T_ground"},
                "T_sink": {"unit": "degC", "range": [25.0, 65.0]},
                "part_load_ratio": {"unit": "-", "range": [0.0, 1.0], "default": 1.0},
            },
            "outputs": {
                "cop": {"unit": "dimensionless"},
                "heating_capacity_kw": {"unit": "kW"},
                "electrical_input_kw": {"unit": "kW"},
                "T_ground": {"unit": "degC"},
            },
            "source": "Staffell et al. (2012); ASHRAE (2019) Ch.34; EN 14825:2016",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for m in range(1, 13):
        r = model.predict({"month": m, "T_sink": 35.0, "part_load_ratio": 0.7})
        print(f"Month {m:2d}: T_ground={float(r['T_ground']):.1f}C, COP={float(r['cop']):.2f}")
