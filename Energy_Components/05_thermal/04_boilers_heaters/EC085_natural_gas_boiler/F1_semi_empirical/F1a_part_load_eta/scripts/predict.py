"""
EC085 — Natural Gas Boiler — F1a Part-Load Efficiency
ComponentModel wrapper with standardised predict() / get_info() interface.
"""

import json
import os
from model import NaturalGasBoilerModel

_PARAMS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "parameters.json"
)


def _load_default_params() -> dict:
    with open(_PARAMS_PATH, "r") as fh:
        return json.load(fh)["default_parameters"]


class ComponentModel:
    """
    Standardised wrapper for the Natural Gas Boiler part-load efficiency model.

    Usage
    -----
    >>> model = ComponentModel()
    >>> out = model.predict({"part_load_ratio": 0.6, "supply_temp": 55.0})
    >>> print(out)
    """

    component_id   = "EC085"
    component_name = "Natural Gas Boiler (Condensing)"
    fidelity       = "F1a — Part-Load Efficiency"
    version        = "1.0.0"

    def __init__(self, params: dict = None):
        defaults = _load_default_params()
        if params:
            defaults.update(params)
        self._params  = defaults
        self._physics = NaturalGasBoilerModel(defaults)

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def predict(self, inputs: dict) -> dict:
        """
        Predict boiler operating point.

        Parameters
        ----------
        inputs : dict
            part_load_ratio : float — PLR [0, 1]
            supply_temp     : float — supply water temperature [°C], (30-80)

        Returns
        -------
        dict
            thermal_output_kw, fuel_input_kw, efficiency, gas_consumption_m3h
        """
        PLR   = float(inputs["part_load_ratio"])
        T_sup = float(inputs.get("supply_temp", 60.0))

        if not (0.0 <= PLR <= 1.0):
            raise ValueError(f"part_load_ratio must be in [0, 1], got {PLR}")
        if not (30.0 <= T_sup <= 80.0):
            raise ValueError(f"supply_temp must be in [30, 80] °C, got {T_sup}")

        result = self._physics.evaluate(PLR, T_sup)

        return {
            "thermal_output_kw":   round(result["thermal_output_kw"],   4),
            "fuel_input_kw":       round(result["fuel_input_kw"],       4),
            "efficiency":          round(result["efficiency"],           6),
            "gas_consumption_m3h": round(result["gas_consumption_m3h"], 6),
            "PLR_effective":       round(result["PLR_effective"],        4),
            "condensing_factor":   round(result["condensing_factor"],    4),
        }

    def get_info(self) -> dict:
        """Return component metadata and parameter summary."""
        return {
            "component_id":   self.component_id,
            "component_name": self.component_name,
            "fidelity":       self.fidelity,
            "version":        self.version,
            "inputs": {
                "part_load_ratio": {"unit": "-",  "range": [0.0, 1.0]},
                "supply_temp":     {"unit": "°C", "range": [30.0, 80.0]},
            },
            "outputs": {
                "thermal_output_kw":   "kW",
                "fuel_input_kw":       "kW (thermal input from gas)",
                "efficiency":          "-",
                "gas_consumption_m3h": "m³/h",
                "PLR_effective":       "-",
                "condensing_factor":   "-",
            },
            "active_parameters": self._params,
            "source": (
                "EnergyPlus Engineering Reference (2023), Boiler:HotWater; "
                "Stafford (2009), Energy and Buildings, 41(2), 168-175."
            ),
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(model.get_info())
    print(model.predict({"part_load_ratio": 0.6, "supply_temp": 55.0}))
