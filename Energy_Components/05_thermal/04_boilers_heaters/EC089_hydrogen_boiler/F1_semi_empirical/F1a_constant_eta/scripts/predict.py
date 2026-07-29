"""
EC089 — Hydrogen Boiler — F1a Constant Efficiency
ComponentModel wrapper with standardised predict() / get_info() interface.
"""

import json
import os
from model import HydrogenBoilerModel

_PARAMS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "parameters.json"
)


def _load_default_params() -> dict:
    with open(_PARAMS_PATH, "r") as fh:
        return json.load(fh)["default_parameters"]


class ComponentModel:
    """Standardised wrapper for the Hydrogen Boiler constant-eta model."""

    component_id   = "EC089"
    component_name = "Hydrogen Boiler (100% H2 Combustion)"
    fidelity       = "F1a — Constant Efficiency"
    version        = "1.0.0"

    def __init__(self, params: dict = None):
        defaults = _load_default_params()
        if params:
            defaults.update(params)
        self._params  = defaults
        self._physics = HydrogenBoilerModel(defaults)

    def predict(self, inputs: dict) -> dict:
        """
        Parameters
        ----------
        inputs : dict
            part_load_ratio : float — PLR [0, 1]
        """
        PLR = float(inputs["part_load_ratio"])
        if not (0.0 <= PLR <= 1.0):
            raise ValueError(f"part_load_ratio must be in [0, 1], got {PLR}")

        result = self._physics.evaluate(PLR)

        return {
            "thermal_output_kw":           round(result["thermal_output_kw"],   4),
            "fuel_input_kw":               round(result["fuel_input_kw"],       4),
            "efficiency":                  round(result["efficiency"],           6),
            "h2_mass_flow_kg_h":           round(result["h2_mass_flow_kg_h"],   6),
            "water_vapour_kg_h":           round(result["water_vapour_kg_h"],   6),
            "PLR_effective":               round(result["PLR_effective"],        4),
            "standby_power_kw":            result["standby_power_kw"],
            "co2_emissions_g_per_kwh_th":  result["co2_emissions_g_per_kwh_th"],
        }

    def get_info(self) -> dict:
        return {
            "component_id":   self.component_id,
            "component_name": self.component_name,
            "fidelity":       self.fidelity,
            "version":        self.version,
            "inputs": {
                "part_load_ratio": {"unit": "-", "range": [0.0, 1.0]},
            },
            "outputs": {
                "thermal_output_kw":          "kW",
                "fuel_input_kw":              "kW (thermal input from H2)",
                "efficiency":                 "-",
                "h2_mass_flow_kg_h":          "kg/h",
                "water_vapour_kg_h":          "kg/h",
                "PLR_effective":              "-",
                "standby_power_kw":           "kW",
                "co2_emissions_g_per_kwh_th": "g/kWh_th (point of use, zero)",
            },
            "active_parameters": self._params,
            "source": (
                "Hy4Heat WP6 (2021); BEIS UK Hydrogen Heating Trials (2022); "
                "Cellek & Pinarbasi (2018) Int. J. Hydrogen Energy 43."
            ),
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(model.get_info())
    print(model.predict({"part_load_ratio": 0.6}))
