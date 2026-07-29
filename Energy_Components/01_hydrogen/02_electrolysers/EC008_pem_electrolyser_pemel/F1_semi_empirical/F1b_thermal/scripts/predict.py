"""
EC008 -- PEM Electrolyser (PEMEL) -- F1b Thermal
ComponentModel wrapper with standardised predict() / get_info() interface.
"""

import json
import os
from model import PEMELThermalModel

_PARAMS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "parameters.json"
)


def _load_default_params() -> dict:
    with open(_PARAMS_PATH, "r") as fh:
        return json.load(fh)["default_parameters"]


class ComponentModel:
    component_id   = "EC008"
    component_name = "PEM Electrolyser (PEMEL)"
    fidelity       = "F1b -- Thermal (temperature-dependent V-I)"
    version        = "1.0.0"

    def __init__(self, params: dict = None):
        defaults = _load_default_params()
        if params:
            defaults.update(params)
        self._params = defaults
        self._physics = PEMELThermalModel(defaults)

    def predict(self, inputs: dict) -> dict:
        """
        Parameters
        ----------
        inputs : dict
            current_density : float -- A/cm2
            temperature     : float -- K (323-363)
            pressure        : float -- bar (optional)
        """
        j = float(inputs["current_density"])
        T = float(inputs["temperature"])

        if not (0.0 <= j <= 4.0):
            raise ValueError(f"current_density must be in [0, 4.0] A/cm2, got {j}")
        if not (300.0 <= T <= 400.0):
            raise ValueError(f"temperature must be in [300, 400] K, got {T}")

        result = self._physics.evaluate(j, T)

        return {
            "cell_voltage_V":          float(round(float(result["cell_voltage"]), 6)),
            "power_consumption_W_cm2": float(round(float(result["power_consumption"]), 6)),
            "efficiency_voltage":      float(round(float(result["efficiency_voltage"]), 6)),
            "efficiency_faradaic":     float(round(float(result["efficiency_faradaic"]), 6)),
            "h2_production_rate_mol_s_cm2": float(round(float(result["h2_production_rate"]), 10)),
            "heat_generation_W_cm2":   float(round(float(result["heat_generation"]), 6)),
            "E_rev_V":                 float(round(float(result["E_rev"]), 6)),
            "E_tn_V":                  float(round(float(result["E_tn"]), 6)),
            "V_act_anode_V":           float(round(float(result["V_act_anode"]), 6)),
            "V_act_cathode_V":         float(round(float(result["V_act_cathode"]), 6)),
            "V_ohm_V":                 float(round(float(result["V_ohm"]), 6)),
            "membrane_resistance_ohm_cm2": float(round(float(result["membrane_resistance"]), 6)),
        }

    def get_info(self) -> dict:
        return {
            "component_id":   self.component_id,
            "component_name": self.component_name,
            "fidelity":       self.fidelity,
            "version":        self.version,
            "inputs": {
                "current_density": {"unit": "A/cm2", "range": [0.0, 3.0]},
                "temperature":     {"unit": "K",     "range": [323.0, 363.0]},
                "pressure":        {"unit": "bar",   "range": [1.0, 80.0], "optional": True},
            },
            "outputs": {
                "cell_voltage_V":          "V",
                "power_consumption_W_cm2": "W/cm2",
                "efficiency_voltage":      "-",
                "efficiency_faradaic":     "-",
                "h2_production_rate_mol_s_cm2": "mol/s/cm2",
                "heat_generation_W_cm2":   "W/cm2",
            },
            "active_parameters": self._params,
            "source": "Garcia-Valverde et al. (2012); Springer et al. (1991)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(model.get_info())
    print(model.predict({"current_density": 1.0, "temperature": 353.15}))
