"""EC090 — Solar Water Heater Combi-System — F1b Part-Load — Standardized Interface"""
import json, numpy as np
from pathlib import Path
from model import SolarWaterHeaterF1b

_PARAMS_PATH = Path(__file__).parent.parent / "data" / "parameters.json"


def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)["default_parameters"]


class ComponentModel:
    component_id   = "EC090"
    component_name = "Solar Water Heater Combi-System"
    fidelity       = "F1b"

    def __init__(self, params: dict = None):
        defaults = _load_params()
        if params:
            defaults.update(params)
        self._params  = defaults
        self._physics = SolarWaterHeaterF1b(defaults)

    def predict(self, inputs: dict) -> dict:
        I     = np.asarray(inputs["I_solar_w_m2"], dtype=float)
        Q_dem = np.asarray(inputs["Q_demand_kw"],  dtype=float)
        T_amb = inputs.get("T_ambient", None)
        if T_amb is not None:
            T_amb = np.asarray(T_amb, dtype=float)
        return self._physics.evaluate(I, Q_dem, T_amb)

    def get_info(self) -> dict:
        return {
            "name":        "Solar Water Heater Combi-System",
            "ec_id":       "EC090",
            "fidelity":    "F1b",
            "description": "HWB collector efficiency + auxiliary boiler part-load curve + tank standby",
            "inputs": {
                "I_solar_w_m2": {"unit": "W/m2", "range": [0, 1200]},
                "Q_demand_kw":  {"unit": "kW",   "range": [0, 50]},
                "T_ambient":    {"unit": "degC",  "range": [-20, 40], "default": 10},
            },
            "outputs": {
                "Q_solar_kw":     {"unit": "kW"},
                "Q_aux_kw":       {"unit": "kW"},
                "Q_aux_fuel_kw":  {"unit": "kW"},
                "Q_standby_kw":   {"unit": "kW"},
                "solar_fraction": {"unit": "-", "range": [0, 1]},
                "eta_collector":  {"unit": "-"},
                "eta_aux":        {"unit": "-"},
                "PLR_aux":        {"unit": "-"},
            },
            "source": "EN 12975:2006; Duffie & Beckman (2013); Haller (2013)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for I in [0, 200, 400, 600, 800, 1000]:
        r = model.predict({"I_solar_w_m2": float(I), "Q_demand_kw": 15.0, "T_ambient": 10.0})
        print(f"I={I:4.0f} W/m2: Q_solar={float(r['Q_solar_kw']):.2f}kW, "
              f"SF={float(r['solar_fraction']):.3f}, "
              f"eta_coll={float(r['eta_collector']):.3f}, "
              f"Q_aux={float(r['Q_aux_kw']):.2f}kW")
