"""EC081 — Thermochemical TES — F1b Stratified Multi-Node — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import ThermochemicalTESF1b

_PARAMS_PATH = Path(__file__).parent.parent / "data" / "parameters.json"


def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)["default_parameters"]


class ComponentModel:
    component_id   = "EC081"
    component_name = "Thermochemical Energy Storage"
    fidelity       = "F1b"

    def __init__(self, params: dict = None):
        defaults = _load_params()
        if params:
            defaults["unit"].update(params.get("unit", {}))
        self._params = defaults
        self._model  = ThermochemicalTESF1b(defaults)

    def predict(self, inputs: dict) -> dict:
        q_ch  = float(inputs.get("q_charge_W",    0.0))
        q_dis = float(inputs.get("q_discharge_W", 0.0))
        T_amb = float(inputs.get("t_ambient",     self._model.T_amb_default))
        dur   = float(inputs.get("duration_s",    3600.0))
        dt    = float(inputs.get("dt",            60.0))

        x_init = inputs.get("x_initial", None)
        if x_init is not None and not np.isscalar(x_init):
            x_init = np.asarray(x_init, dtype=float)

        return self._model.simulate(q_ch, q_dis, T_amb, dur, x_init, dt)

    def get_info(self) -> dict:
        return {
            "name":        "Thermochemical Energy Storage",
            "ec_id":       "EC081",
            "fidelity":    "F1b",
            "description": "Multi-node reaction-front tracking with thermal losses and η_rt on charge side",
            "inputs": {
                "q_charge_W":    {"unit": "W",    "range": [0, 200000]},
                "q_discharge_W": {"unit": "W",    "range": [0, 150000]},
                "t_ambient":     {"unit": "degC", "range": [-20, 50]},
                "duration_s":    {"unit": "s",    "range": [60, 86400]},
                "x_initial":     {"unit": "-",    "range": [0, 1], "default": 0},
                "dt":            {"unit": "s",    "default": 60},
            },
            "outputs": {
                "x_nodes":               {"unit": "-",  "description": "Reaction extent per node [0=hydrated, 1=dehydrated]"},
                "soc":                   {"unit": "-",  "range": [0, 1]},
                "Q_actual_charge_kw":    {"unit": "kW"},
                "Q_actual_discharge_kw": {"unit": "kW"},
                "Q_loss_kw":             {"unit": "kW"},
                "stratification_index":  {"unit": "-",  "range": [0, 1]},
                "x_history":             {"unit": "-",  "description": "Array (n_steps+1, N)"},
            },
            "source": "Kerskes (2012); N'Tsoukpoe (2009); Tatsidjodoung (2013)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"q_charge_W": 100000.0, "q_discharge_W": 0.0,
                       "t_ambient": 20.0, "duration_s": 3600.0})
    print(f"SOC after 1h charge: {r['soc']:.3f}")
    print(f"Q_actual_charge: {r['Q_actual_charge_kw']:.2f} kW")
    print(f"Q_loss: {r['Q_loss_kw']:.3f} kW")
    print(f"Stratification index: {r['stratification_index']:.3f}")
