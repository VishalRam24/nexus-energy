"""
EC086 — Electric Boiler / Resistance Heater — F1a Constant Efficiency
ComponentModel wrapper with standardised predict() / get_info() interface.
"""

import json
import os
from model import ElectricBoilerModel

_PARAMS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "parameters.json"
)


def _load_default_params() -> dict:
    with open(_PARAMS_PATH, "r") as fh:
        return json.load(fh)["default_parameters"]


class ComponentModel:
    """
    Standardised wrapper for the Electric Boiler / Resistance Heater
    constant-efficiency model.

    Usage
    -----
    >>> model = ComponentModel()
    >>> out = model.predict({"part_load_ratio": 0.6})
    """

    component_id   = "EC086"
    component_name = "Electric Boiler / Resistance Heater"
    fidelity       = "F1a — Constant Efficiency"
    version        = "1.0.0"

    def __init__(self, params: dict = None):
        defaults = _load_default_params()
        if params:
            defaults.update(params)
        self._params  = defaults
        self._physics = ElectricBoilerModel(defaults)

    # ------------------------------------------------------------------

    def predict(self, inputs: dict) -> dict:
        """
        Predict electric boiler operating point.

        Parameters
        ----------
        inputs : dict
            part_load_ratio : float — PLR [0, 1]

        Returns
        -------
        dict
            thermal_output_kw, electrical_input_kw, efficiency,
            PLR_effective, co2_emissions_g_per_kwh_th
        """
        PLR = float(inputs["part_load_ratio"])
        if not (0.0 <= PLR <= 1.0):
            raise ValueError(f"part_load_ratio must be in [0, 1], got {PLR}")

        result = self._physics.evaluate(PLR)

        return {
            "thermal_output_kw":           round(result["thermal_output_kw"],   6),
            "electrical_input_kw":         round(result["electrical_input_kw"], 6),
            "efficiency":                  round(result["efficiency"],           6),
            "PLR_effective":               round(result["PLR_effective"],        6),
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
                "electrical_input_kw":        "kW",
                "efficiency":                 "-",
                "PLR_effective":              "-",
                "co2_emissions_g_per_kwh_th": "g/kWh_th (point-of-use)",
            },
            "active_parameters": self._params,
            "source": (
                "ASHRAE Handbook (HVAC Systems & Equipment, 2020), Ch. 32 "
                "'Boilers'; IEA Task 44 reference electric heater."
            ),
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(model.get_info())
    print(model.predict({"part_load_ratio": 0.6}))
