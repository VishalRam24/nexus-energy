"""
EC003 -- Alkaline Fuel Cell (AFC) -- F1a Polarization Curve
ComponentModel wrapper with standardised predict() / get_info() interface.
"""

import json, os
from model import AFCModel

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_default_params() -> dict:
    with open(_PARAMS_PATH, "r") as fh:
        return json.load(fh)["default_parameters"]


class ComponentModel:
    component_id   = "EC003"
    component_name = "Alkaline Fuel Cell (AFC)"
    fidelity       = "F1a -- Polarization Curve"
    version        = "1.0.0"

    def __init__(self, params: dict = None):
        defaults = _load_default_params()
        if params:
            defaults.update(params)
        self._params  = defaults
        self._physics = AFCModel(defaults)

    def predict(self, inputs: dict) -> dict:
        j   = float(inputs["current_density"])
        T_C = float(inputs.get("temperature", self._params["T"] - 273.15))

        j_L = self._physics.j_L
        if not (0.0 <= j < j_L):
            raise ValueError(f"current_density must be in [0, {j_L}) A/cm2, got {j}")

        result = self._physics.evaluate(j, T_C)
        return {k: round(float(v), 6) for k, v in result.items()}

    def get_info(self) -> dict:
        return {
            "component_id":   self.component_id,
            "component_name": self.component_name,
            "fidelity":       self.fidelity,
            "version":        self.version,
            "inputs": {
                "current_density": {"unit": "A/cm2", "range": [0.0, self._physics.j_L]},
                "temperature":     {"unit": "degC",  "range": [50.0, 90.0]},
            },
            "outputs": {
                "cell_voltage_V": "V", "stack_voltage_V": "V",
                "power_density_W_cm2": "W/cm2", "stack_power_W": "W",
                "efficiency": "-", "E_Nernst_V": "V",
                "V_act_V": "V", "V_ohm_V": "V", "V_conc_V": "V",
            },
            "active_parameters": self._params,
            "source": "Larminie & Dicks (2003); Appleby & Foulkes (1989)",
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print(m.predict({"current_density": 0.3, "temperature": 70.0}))
