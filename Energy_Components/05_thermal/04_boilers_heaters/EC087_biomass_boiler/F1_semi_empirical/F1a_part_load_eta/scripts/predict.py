"""
EC087 — Biomass Boiler — F1a Part-Load Efficiency
ComponentModel wrapper with standardised predict() / get_info() interface.
"""

import json
import os
from model import BiomassBoilerModel

_PARAMS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "parameters.json"
)


def _load_default_params() -> dict:
    with open(_PARAMS_PATH, "r") as fh:
        return json.load(fh)["default_parameters"]


class ComponentModel:
    """
    Standardised wrapper for the Biomass Boiler part-load efficiency model.
    """

    component_id   = "EC087"
    component_name = "Biomass Boiler (Wood Pellet / Wood Chip)"
    fidelity       = "F1a — Part-Load Efficiency"
    version        = "1.0.0"

    def __init__(self, params: dict = None):
        defaults = _load_default_params()
        if params:
            defaults.update(params)
        self._params  = defaults
        self._physics = BiomassBoilerModel(defaults)

    def predict(self, inputs: dict) -> dict:
        """
        Parameters
        ----------
        inputs : dict
            part_load_ratio : float — PLR [0, 1]

        Returns
        -------
        dict : thermal_output_kw, fuel_input_kw, efficiency,
               fuel_mass_flow_kg_h, LHV_effective_MJ_kg, ...
        """
        PLR = float(inputs["part_load_ratio"])
        if not (0.0 <= PLR <= 1.0):
            raise ValueError(f"part_load_ratio must be in [0, 1], got {PLR}")

        result = self._physics.evaluate(PLR)

        return {
            "thermal_output_kw":           round(result["thermal_output_kw"],   4),
            "fuel_input_kw":               round(result["fuel_input_kw"],       4),
            "efficiency":                  round(result["efficiency"],           6),
            "fuel_mass_flow_kg_h":         round(result["fuel_mass_flow_kg_h"], 6),
            "LHV_effective_MJ_kg":         round(result["LHV_effective_MJ_kg"], 4),
            "PLR_effective":               round(result["PLR_effective"],        4),
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
                "fuel_input_kw":              "kW (thermal input from biomass)",
                "efficiency":                 "-",
                "fuel_mass_flow_kg_h":        "kg/h (as-received fuel)",
                "LHV_effective_MJ_kg":        "MJ/kg",
                "PLR_effective":              "-",
                "co2_emissions_g_per_kwh_th": "g/kWh_th (non-biogenic)",
            },
            "active_parameters": self._params,
            "source": (
                "EN 303-5:2012; IEA Bioenergy Task 32; "
                "Carvalho et al. (2013) Energy 58, 290-301."
            ),
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(model.get_info())
    print(model.predict({"part_load_ratio": 0.6}))
