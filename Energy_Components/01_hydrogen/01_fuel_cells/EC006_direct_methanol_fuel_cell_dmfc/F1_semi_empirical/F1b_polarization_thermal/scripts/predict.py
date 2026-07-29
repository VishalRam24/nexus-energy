"""
EC006 -- Direct Methanol Fuel Cell (DMFC) -- F1b Polarization-Thermal
ComponentModel wrapper with standardised predict() / get_info() interface.
"""

import json
import os
from model import DMFCThermalModel

_PARAMS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "parameters.json"
)


def _load_default_params() -> dict:
    with open(_PARAMS_PATH, "r") as fh:
        return json.load(fh)["default_parameters"]


class ComponentModel:
    """
    Standardised wrapper for the DMFC thermal polarization model.

    Usage
    -----
    >>> model = ComponentModel()
    >>> out = model.predict({"current_density": 0.15, "temperature": 353.15})
    """

    component_id   = "EC006"
    component_name = "Direct Methanol Fuel Cell (DMFC)"
    fidelity       = "F1b -- Polarization-Thermal (crossover current, Nafion conductivity, Arrhenius anode kinetics)"
    version        = "1.0.0"

    def __init__(self, params: dict = None):
        defaults = _load_default_params()
        if params:
            defaults.update(params)
        self._params = defaults
        self._physics = DMFCThermalModel(defaults)

    def predict(self, inputs: dict) -> dict:
        """
        Predict DMFC operating point.

        Parameters
        ----------
        inputs : dict
            current_density : float -- A/cm2
            temperature     : float -- K (333-383)

        Returns
        -------
        dict
        """
        j = float(inputs["current_density"])
        T = float(inputs["temperature"])

        if not (0.0 <= j < self._physics.j_L):
            raise ValueError(
                f"current_density must be in [0, {self._physics.j_L}) A/cm2, got {j}"
            )
        if not (300.0 <= T <= 430.0):
            raise ValueError(f"temperature must be in [300, 430] K, got {T}")

        result = self._physics.evaluate(j, T)

        return {
            "cell_voltage_V":            float(round(float(result["cell_voltage"]), 6)),
            "power_density_W_cm2":       float(round(float(result["power_density"]), 6)),
            "efficiency":                float(round(float(result["efficiency"]), 6)),
            "heat_generation_W_cm2":     float(round(float(result["heat_generation"]), 6)),
            "membrane_resistance_ohm_cm2": float(round(float(result["membrane_resistance"]), 6)),
            "crossover_current_A_cm2":   float(round(float(result["crossover_current"]), 6)),
            "E_nernst_V":                float(round(float(result["E_nernst"]), 6)),
            "V_act_V":                   float(round(float(result["V_act"]), 6)),
            "V_mix_V":                   float(round(float(result["V_mix"]), 6)),
            "V_ohm_V":                   float(round(float(result["V_ohm"]), 6)),
            "V_conc_V":                  float(round(float(result["V_conc"]), 6)),
        }

    def get_info(self) -> dict:
        return {
            "component_id":   self.component_id,
            "component_name": self.component_name,
            "fidelity":       self.fidelity,
            "version":        self.version,
            "inputs": {
                "current_density": {"unit": "A/cm2", "range": [0.0, self._physics.j_L]},
                "temperature":     {"unit": "K",     "range": [333.0, 383.0]},
            },
            "outputs": {
                "cell_voltage_V":              "V",
                "power_density_W_cm2":         "W/cm2",
                "efficiency":                  "-",
                "heat_generation_W_cm2":       "W/cm2",
                "membrane_resistance_ohm_cm2": "ohm cm2",
                "crossover_current_A_cm2":     "A/cm2",
                "V_mix_V":                     "V (mixed-potential crossover loss)",
            },
            "active_parameters": self._params,
            "source": "Scott et al. (1999); Kulikovsky (2002); Nordlund & Lindbergh (2002)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(model.get_info())
    print(model.predict({"current_density": 0.15, "temperature": 353.15}))
