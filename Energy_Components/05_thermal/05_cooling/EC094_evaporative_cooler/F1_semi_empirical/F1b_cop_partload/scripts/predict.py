"""EC094 — Evaporative Cooler — F1b COP/EER Part-Load — Standardized Interface"""
import json, numpy as np
from pathlib import Path
from model import EvaporativeCoolerF1b

_PARAMS_PATH = Path(__file__).parent.parent / "data" / "parameters.json"


def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)["default_parameters"]


class ComponentModel:
    component_id   = "EC094"
    component_name = "Evaporative Cooler"
    fidelity       = "F1b"

    def __init__(self, params: dict = None):
        defaults = _load_params()
        if params:
            defaults.update(params)
        self._params  = defaults
        self._physics = EvaporativeCoolerF1b(defaults)

    def predict(self, inputs: dict) -> dict:
        T_db = np.asarray(inputs["T_db_c"],  dtype=float)
        RH   = np.asarray(inputs["RH_pct"],  dtype=float)
        plr  = np.asarray(inputs.get("PLR", 1.0), dtype=float)
        return self._physics.evaluate(T_db, RH, plr)

    def get_info(self) -> dict:
        return {
            "name":        "Evaporative Cooler",
            "ec_id":       "EC094",
            "fidelity":    "F1b",
            "description": "Saturation effectiveness + DOE-2 part-load EER + humidity correction",
            "inputs": {
                "T_db_c":  {"unit": "degC", "range": [10, 55]},
                "RH_pct":  {"unit": "%",    "range": [5, 95]},
                "PLR":     {"unit": "-",    "range": [0.2, 1.0], "default": 1.0},
            },
            "outputs": {
                "Q_cool_kw":  {"unit": "kW"},
                "W_fan_kw":   {"unit": "kW"},
                "EER":        {"unit": "kW/kW"},
                "T_wb_c":     {"unit": "degC"},
                "T_outlet_c": {"unit": "degC"},
                "f_humidity": {"unit": "-"},
                "f_partload": {"unit": "-"},
            },
            "source": "ASHRAE Fundamentals (2021); Watt & Brown (1997); Stull (2011)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for T_db, RH in [(35, 20), (35, 40), (35, 60), (40, 30), (25, 50)]:
        r = model.predict({"T_db_c": float(T_db), "RH_pct": float(RH)})
        print(f"T_db={T_db}C, RH={RH}%: Q={float(r['Q_cool_kw']):.1f}kW, "
              f"EER={float(r['EER']):.1f}, T_out={float(r['T_outlet_c']):.1f}C, "
              f"T_wb={float(r['T_wb_c']):.1f}C")
