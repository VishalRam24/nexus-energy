"""EC088 — Oil-Fired Boiler — F1b Part-Load — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import OilBoilerF1b

_PARAMS_PATH = Path(__file__).parent.parent / "data" / "parameters.json"


def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)["default_parameters"]


class ComponentModel:
    component_id   = "EC088"
    component_name = "Oil-Fired Boiler"
    fidelity       = "F1b"

    def __init__(self, params: dict = None):
        defaults = _load_params()
        if params:
            defaults.update(params)
        self._params  = defaults
        self._physics = OilBoilerF1b(defaults)

    def predict(self, inputs: dict) -> dict:
        PLR       = np.asarray(inputs["PLR"], dtype=float)
        flue_temp = inputs.get("flue_gas_temp", None)
        if flue_temp is not None:
            flue_temp = np.asarray(flue_temp, dtype=float)
        result = self._physics.evaluate(PLR, flue_temp)
        return {
            "efficiency":      result["efficiency"],
            "heat_output_kw":  result["heat_output_kw"],
            "fuel_input_kw":   result["fuel_input_kw"],
            "flue_loss_kw":    result["flue_loss_kw"],
            "standby_loss_kw": result["standby_loss_kw"],
            "flue_gas_temp_c": result["flue_gas_temp_c"],
        }

    def get_info(self) -> dict:
        return {
            "name":        "Oil-Fired Boiler",
            "ec_id":       "EC088",
            "fidelity":    "F1b",
            "description": "eta(PLR)=a0+a1*PLR+a2*PLR^2 with oil-specific flue and standby losses",
            "inputs": {
                "PLR":           {"unit": "-",    "range": [0.0, 1.0]},
                "flue_gas_temp": {"unit": "degC", "range": [80, 300], "default": "auto"},
            },
            "outputs": {
                "efficiency":      {"unit": "-"},
                "heat_output_kw":  {"unit": "kW"},
                "fuel_input_kw":   {"unit": "kW"},
                "flue_loss_kw":    {"unit": "kW"},
                "standby_loss_kw": {"unit": "kW"},
                "flue_gas_temp_c": {"unit": "degC"},
            },
            "source": "EN 303-1:2017; ASHRAE HVAC S&E (2020); EnergyPlus (2023)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for plr in [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]:
        r = model.predict({"PLR": plr})
        print(f"PLR={plr:.1f}: eta={float(r['efficiency']):.3f}, "
              f"Q_out={float(r['heat_output_kw']):.1f}kW, "
              f"Q_flue={float(r['flue_loss_kw']):.2f}kW, "
              f"T_flue={float(r['flue_gas_temp_c']):.0f}degC")
