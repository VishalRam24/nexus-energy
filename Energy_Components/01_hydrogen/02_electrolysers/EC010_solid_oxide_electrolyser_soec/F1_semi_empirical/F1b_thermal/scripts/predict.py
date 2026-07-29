"""
EC010 -- Solid Oxide Electrolyser (SOEC) -- F1b Thermal
ComponentModel wrapper with standardised predict() / get_info() interface.
"""

import json
import os
import numpy as np
from model import SOECThermalModel

_PARAMS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "parameters.json"
)


def _load_default_params() -> dict:
    with open(_PARAMS_PATH, "r") as fh:
        return json.load(fh)["default_parameters"]


_THERMAL_MODE_MAP = {1: "endothermic", 0: "thermoneutral", -1: "exothermic"}


class ComponentModel:
    component_id   = "EC010"
    component_name = "Solid Oxide Electrolyser (SOEC)"
    fidelity       = "F1b -- Thermal (temperature-dependent ASR, activation, thermal mode)"
    version        = "1.0.0"

    def __init__(self, params: dict = None):
        defaults = _load_default_params()
        if params:
            defaults.update(params)
        self._params = defaults
        self._physics = SOECThermalModel(defaults)

    def predict(self, inputs: dict) -> dict:
        """
        Parameters
        ----------
        inputs : dict
            current_density   : float -- A/cm2
            temperature       : float -- K (973-1123)
            steam_utilization : float -- [0-0.8] (optional)
        """
        j = float(inputs["current_density"])
        T = float(inputs["temperature"])

        if not (0.0 <= j <= 3.0):
            raise ValueError(f"current_density must be in [0, 3.0] A/cm2, got {j}")
        if not (900.0 <= T <= 1200.0):
            raise ValueError(f"temperature must be in [900, 1200] K, got {T}")

        result = self._physics.evaluate(j, T)

        mode_num = int(result["thermal_mode"])
        mode_str = _THERMAL_MODE_MAP.get(mode_num, "thermoneutral")

        return {
            "cell_voltage_V":          float(round(float(result["cell_voltage"]), 6)),
            "power_consumption_W_cm2": float(round(float(result["power_consumption"]), 6)),
            "efficiency":              float(round(float(result["efficiency"]), 6)),
            "h2_production_rate_mol_s_cm2": float(round(float(result["h2_production_rate"]), 10)),
            "thermal_mode":            mode_str,
            "heat_generation_W_cm2":   float(round(float(result["heat_generation"]), 6)),
            "E_rev_V":                 float(round(float(result["E_rev"]), 6)),
            "E_tn_V":                  float(round(float(result["E_tn"]), 6)),
            "V_act_V":                 float(round(float(result["V_act"]), 6)),
            "V_ohm_V":                 float(round(float(result["V_ohm"]), 6)),
            "ohmic_asr_ohm_cm2":       float(round(float(result["ohmic_asr"]), 6)),
        }

    def get_info(self) -> dict:
        return {
            "component_id":   self.component_id,
            "component_name": self.component_name,
            "fidelity":       self.fidelity,
            "version":        self.version,
            "inputs": {
                "current_density":   {"unit": "A/cm2", "range": [0.0, 2.5]},
                "temperature":       {"unit": "K",     "range": [973.0, 1123.0]},
                "steam_utilization": {"unit": "-",     "range": [0.0, 0.8], "optional": True},
            },
            "outputs": {
                "cell_voltage_V":          "V",
                "power_consumption_W_cm2": "W/cm2",
                "efficiency":              "-",
                "h2_production_rate_mol_s_cm2": "mol/s/cm2",
                "thermal_mode":            "endothermic/thermoneutral/exothermic",
            },
            "active_parameters": self._params,
            "source": "Ni et al. (2007); Udagawa et al. (2007)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(model.get_info())
    print(model.predict({"current_density": 0.5, "temperature": 1073.0}))
