"""
EC009 -- Alkaline Electrolyser (AEL) -- F1b Thermal
ComponentModel wrapper with standardised predict() / get_info() interface.
"""

import json
import os
from model import AELThermalModel

_PARAMS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "parameters.json"
)


def _load_default_params() -> dict:
    with open(_PARAMS_PATH, "r") as fh:
        return json.load(fh)["default_parameters"]


class ComponentModel:
    component_id   = "EC009"
    component_name = "Alkaline Electrolyser (AEL)"
    fidelity       = "F1b -- Thermal (temperature-dependent KOH conductivity and kinetics)"
    version        = "1.0.0"

    def __init__(self, params: dict = None):
        defaults = _load_default_params()
        if params:
            defaults.update(params)
        self._params = defaults
        self._physics = AELThermalModel(defaults)

    def predict(self, inputs: dict) -> dict:
        """
        Parameters
        ----------
        inputs : dict
            current_density   : float -- A/m2
            temperature       : float -- K (333-373)
            koh_concentration : float -- wt% (optional, default 30)
        """
        j = float(inputs["current_density"])
        T = float(inputs["temperature"])

        if not (0.0 <= j <= 6000.0):
            raise ValueError(f"current_density must be in [0, 6000] A/m2, got {j}")
        if not (300.0 <= T <= 400.0):
            raise ValueError(f"temperature must be in [300, 400] K, got {T}")

        if "koh_concentration" in inputs:
            self._physics.koh_conc = float(inputs["koh_concentration"])

        result = self._physics.evaluate(j, T)

        return {
            "cell_voltage_V":          float(round(float(result["cell_voltage"]), 6)),
            "power_consumption_kW":    float(round(float(result["power_consumption"]), 6)),
            "efficiency":              float(round(float(result["efficiency"]), 6)),
            "h2_production_rate_mol_s": float(round(float(result["h2_production_rate"]), 8)),
            "E_rev_V":                 float(round(float(result["E_rev"]), 6)),
            "V_act_V":                 float(round(float(result["V_act"]), 6)),
            "V_ohm_V":                 float(round(float(result["V_ohm"]), 6)),
            "koh_conductivity_S_cm":   float(round(float(result["koh_conductivity"]), 6)),
            "bubble_coverage":         float(round(float(result["bubble_coverage"]), 6)),
        }

    def get_info(self) -> dict:
        return {
            "component_id":   self.component_id,
            "component_name": self.component_name,
            "fidelity":       self.fidelity,
            "version":        self.version,
            "inputs": {
                "current_density":   {"unit": "A/m2", "range": [0.0, 5000.0]},
                "temperature":       {"unit": "K",    "range": [333.0, 373.0]},
                "koh_concentration": {"unit": "wt%",  "range": [25.0, 40.0], "optional": True},
            },
            "outputs": {
                "cell_voltage_V":        "V",
                "power_consumption_kW":  "kW",
                "efficiency":            "-",
                "h2_production_rate_mol_s": "mol/s",
            },
            "active_parameters": self._params,
            "source": "Ulleberg (2003); See & White (1997)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(model.get_info())
    print(model.predict({"current_density": 2000.0, "temperature": 353.0}))
