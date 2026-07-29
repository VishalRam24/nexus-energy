"""EC084 — ATES — F1b Stratified Multi-Node — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import AquiferTESF1b

_PARAMS_PATH = Path(__file__).parent.parent / "data" / "parameters.json"


def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)["default_parameters"]


class ComponentModel:
    component_id   = "EC084"
    component_name = "Aquifer Thermal Energy Storage (ATES)"
    fidelity       = "F1b"

    def __init__(self, params: dict = None):
        defaults = _load_params()
        if params:
            defaults["unit"].update(params.get("unit", {}))
        self._params = defaults
        self._model  = AquiferTESF1b(defaults)

    def predict(self, inputs: dict) -> dict:
        q_ch  = float(inputs.get("q_charge_W",    0.0))
        q_dis = float(inputs.get("q_discharge_W", 0.0))
        T_amb = float(inputs.get("t_ambient",     self._model.T_aquifer_natural))
        dur   = float(inputs.get("duration_s",    3600.0))
        dt    = float(inputs.get("dt",            600.0))

        T_init = inputs.get("T_initial", None)
        if T_init is not None and not np.isscalar(T_init):
            T_init = np.asarray(T_init, dtype=float)

        return self._model.simulate(q_ch, q_dis, T_amb, dur, T_init, dt)

    def get_info(self) -> dict:
        return {
            "name":        "Aquifer Thermal Energy Storage (ATES)",
            "ec_id":       "EC084",
            "fidelity":    "F1b",
            "description": "Multi-node aquifer plume with thermal dispersion, losses, and η_rt on charge",
            "inputs": {
                "q_charge_W":    {"unit": "W",    "range": [0, 800000]},
                "q_discharge_W": {"unit": "W",    "range": [0, 600000]},
                "t_ambient":     {"unit": "degC", "range": [-10, 40]},
                "duration_s":    {"unit": "s",    "range": [600, 2592000]},
                "T_initial":     {"unit": "degC", "default": "T_aquifer_natural"},
                "dt":            {"unit": "s",    "default": 600},
            },
            "outputs": {
                "T_nodes":               {"unit": "degC"},
                "soc":                   {"unit": "-", "range": [0, 1]},
                "Q_actual_charge_kw":    {"unit": "kW"},
                "Q_actual_discharge_kw": {"unit": "kW"},
                "Q_loss_kw":             {"unit": "kW"},
                "stratification_index":  {"unit": "-", "range": [0, 1]},
            },
            "source": "Sanner (2003); Bloemendal & Hartog (2018); Dincer & Rosen (2011)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"q_charge_W": 400000.0, "q_discharge_W": 0.0,
                       "t_ambient": 12.0, "duration_s": 3600.0})
    print(f"SOC after 1h charge: {r['soc']:.4f}")
    print(f"T_nodes: {[f'{t:.1f}' for t in r['T_nodes']]} degC")
    print(f"Q_loss: {r['Q_loss_kw']:.3f} kW")
