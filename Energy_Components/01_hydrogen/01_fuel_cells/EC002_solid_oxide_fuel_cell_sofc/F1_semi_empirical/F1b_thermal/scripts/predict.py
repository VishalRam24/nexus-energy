"""
EC002 -- Solid Oxide Fuel Cell (SOFC) -- F1b Thermal
ComponentModel wrapper with standardised predict() / get_info() interface.
"""

import json
import os
from model import SOFCThermalModel

_PARAMS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "parameters.json"
)


def _load_default_params() -> dict:
    with open(_PARAMS_PATH, "r") as fh:
        return json.load(fh)["default_parameters"]


class ComponentModel:
    component_id   = "EC002"
    component_name = "Solid Oxide Fuel Cell (SOFC)"
    fidelity       = "F1b -- Thermal (temperature-dependent ASR and kinetics)"
    version        = "1.0.0"

    def __init__(self, params: dict = None):
        defaults = _load_default_params()
        if params:
            defaults.update(params)
        self._params = defaults
        self._physics = SOFCThermalModel(defaults)

    def predict(self, inputs: dict) -> dict:
        """
        Parameters
        ----------
        inputs : dict
            current_density : float -- A/cm2
            temperature     : float -- K (973-1273)
            fuel_utilization: float -- [0-0.9] (optional)
        """
        j = float(inputs["current_density"])
        T = float(inputs["temperature"])

        if not (0.0 <= j < self._physics.j_L):
            raise ValueError(f"current_density must be in [0, {self._physics.j_L}), got {j}")
        if not (900.0 <= T <= 1350.0):
            raise ValueError(f"temperature must be in [900, 1350] K, got {T}")

        result = self._physics.evaluate(j, T)

        return {
            "cell_voltage_V":      float(round(float(result["cell_voltage"]), 6)),
            "power_density_W_cm2": float(round(float(result["power_density"]), 6)),
            "efficiency":          float(round(float(result["efficiency"]), 6)),
            "asr_ohm_cm2":         float(round(float(result["asr"]), 6)),
            "heat_generation_W_cm2": float(round(float(result["heat_generation"]), 6)),
            "E_nernst_V":          float(round(float(result["E_nernst"]), 6)),
            "V_act_V":             float(round(float(result["V_act"]), 6)),
            "V_ohm_V":             float(round(float(result["V_ohm"]), 6)),
            "V_conc_V":            float(round(float(result["V_conc"]), 6)),
            "ohmic_asr_ohm_cm2":   float(round(float(result["ohmic_asr"]), 6)),
        }

    def get_info(self) -> dict:
        return {
            "component_id":   self.component_id,
            "component_name": self.component_name,
            "fidelity":       self.fidelity,
            "version":        self.version,
            "inputs": {
                "current_density": {"unit": "A/cm2", "range": [0.0, self._physics.j_L]},
                "temperature":     {"unit": "K",     "range": [973.0, 1273.0]},
                "fuel_utilization": {"unit": "-",     "range": [0.0, 0.9], "optional": True},
            },
            "outputs": {
                "cell_voltage_V":        "V",
                "power_density_W_cm2":   "W/cm2",
                "efficiency":            "-",
                "asr_ohm_cm2":           "ohm cm2",
                "heat_generation_W_cm2": "W/cm2",
            },
            "active_parameters": self._params,
            "source": "Chan et al. (2001); Virkar (2005)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(model.get_info())
    print(model.predict({"current_density": 0.5, "temperature": 1073.0}))
