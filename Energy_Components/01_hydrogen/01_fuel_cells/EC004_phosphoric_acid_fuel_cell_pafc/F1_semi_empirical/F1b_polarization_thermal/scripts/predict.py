"""
EC004 -- Phosphoric Acid Fuel Cell (PAFC) -- F1b Polarization-Thermal
ComponentModel wrapper with standardised predict() / get_info() interface.
"""

import json
import os
from model import PAFCThermalModel

_PARAMS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "parameters.json"
)


def _load_default_params() -> dict:
    with open(_PARAMS_PATH, "r") as fh:
        return json.load(fh)["default_parameters"]


class ComponentModel:
    """
    Standardised wrapper for the PAFC thermal polarization model.

    Usage
    -----
    >>> model = ComponentModel()
    >>> out = model.predict({"current_density": 0.3, "temperature": 453.15})
    """

    component_id   = "EC004"
    component_name = "Phosphoric Acid Fuel Cell (PAFC)"
    fidelity       = "F1b -- Polarization-Thermal (H3PO4 conductivity, Arrhenius cathode kinetics)"
    version        = "1.0.0"

    def __init__(self, params: dict = None):
        defaults = _load_default_params()
        if params:
            defaults.update(params)
        self._params = defaults
        self._physics = PAFCThermalModel(defaults)

    def predict(self, inputs: dict) -> dict:
        """
        Predict PAFC operating point.

        Parameters
        ----------
        inputs : dict
            current_density : float -- A/cm2
            temperature     : float -- K (423-483)
            pressure_h2     : float -- atm (optional)
            pressure_o2     : float -- atm (optional)

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
        if not (400.0 <= T <= 550.0):
            raise ValueError(f"temperature must be in [400, 550] K, got {T}")

        if "pressure_h2" in inputs:
            self._physics.pH2 = float(inputs["pressure_h2"])
        if "pressure_o2" in inputs:
            self._physics.pO2 = float(inputs["pressure_o2"])

        result = self._physics.evaluate(j, T)

        return {
            "cell_voltage_V":             float(round(float(result["cell_voltage"]), 6)),
            "power_density_W_cm2":        float(round(float(result["power_density"]), 6)),
            "efficiency":                 float(round(float(result["efficiency"]), 6)),
            "heat_generation_W_cm2":      float(round(float(result["heat_generation"]), 6)),
            "acid_resistance_ohm_cm2":    float(round(float(result["acid_resistance"]), 6)),
            "acid_conductivity_S_cm":     float(round(float(result["acid_conductivity"]), 6)),
            "thermoneutral_voltage_V":    float(round(float(result["thermoneutral_voltage"]), 6)),
            "E_nernst_V":                 float(round(float(result["E_nernst"]), 6)),
            "V_act_V":                    float(round(float(result["V_act"]), 6)),
            "V_ohm_V":                    float(round(float(result["V_ohm"]), 6)),
            "V_conc_V":                   float(round(float(result["V_conc"]), 6)),
        }

    def get_info(self) -> dict:
        return {
            "component_id":   self.component_id,
            "component_name": self.component_name,
            "fidelity":       self.fidelity,
            "version":        self.version,
            "inputs": {
                "current_density": {"unit": "A/cm2", "range": [0.0, self._physics.j_L]},
                "temperature":     {"unit": "K",     "range": [423.0, 483.0]},
                "pressure_h2":     {"unit": "atm",   "range": [0.5, 3.0], "optional": True},
                "pressure_o2":     {"unit": "atm",   "range": [0.1, 1.0], "optional": True},
            },
            "outputs": {
                "cell_voltage_V":          "V",
                "power_density_W_cm2":     "W/cm2",
                "efficiency":              "-",
                "heat_generation_W_cm2":   "W/cm2",
                "acid_resistance_ohm_cm2": "ohm cm2",
                "acid_conductivity_S_cm":  "S/cm",
                "thermoneutral_voltage_V": "V",
            },
            "active_parameters": self._params,
            "source": "Razaq et al. (1989); Appleby & Foulkes (1989); Li et al. (2003)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(model.get_info())
    print(model.predict({"current_density": 0.3, "temperature": 453.15}))
