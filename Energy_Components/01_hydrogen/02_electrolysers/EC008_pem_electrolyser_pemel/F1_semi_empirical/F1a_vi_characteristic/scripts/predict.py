"""
EC008 — PEM Electrolyser (PEMEL) — F1a V-I Characteristic
ComponentModel wrapper with standardised predict() / get_info() interface.
"""

import json
import os
from model import PEMElectrolyserModel

_PARAMS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "parameters.json"
)


def _load_default_params() -> dict:
    with open(_PARAMS_PATH, "r") as fh:
        return json.load(fh)["default_parameters"]


class ComponentModel:
    """
    Standardised wrapper for the PEM Electrolyser V-I model.

    Usage
    -----
    >>> model = ComponentModel()
    >>> out = model.predict({"current_density": 1.0, "temperature": 80.0})
    >>> print(out)
    """

    component_id = "EC008"
    component_name = "PEM Electrolyser (PEMEL)"
    fidelity = "F1a — V-I Characteristic (semi-empirical)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        """
        Parameters
        ----------
        params : dict, optional
            Override any default parameter values.
        """
        defaults = _load_default_params()
        if params:
            defaults.update(params)
        self._params = defaults
        self._physics = PEMElectrolyserModel(defaults)

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def predict(self, inputs: dict) -> dict:
        """
        Predict electrolyser operating point.

        Parameters
        ----------
        inputs : dict
            current_density : float  — A/cm²  (0 – 2.0)
            temperature     : float  — °C     (40 – 90), optional

        Returns
        -------
        dict
            cell_voltage_V, stack_voltage_V, hydrogen_rate_mol_s,
            power_W, efficiency, E_rev_V, V_act_V, V_ohm_V
        """
        j = float(inputs["current_density"])
        T_C = float(inputs.get("temperature", self._params["T"] - 273.15))

        if not (0.0 <= j <= 2.0):
            raise ValueError(f"current_density must be in [0, 2.0] A/cm², got {j}")
        if not (40.0 <= T_C <= 90.0):
            raise ValueError(f"temperature must be in [40, 90] °C, got {T_C}")

        result = self._physics.evaluate(j, T_C)

        return {
            "cell_voltage_V":       round(result["cell_voltage_V"], 6),
            "stack_voltage_V":      round(result["stack_voltage_V"], 4),
            "hydrogen_rate_mol_s":  result["hydrogen_rate_mol_s"],
            "power_W":              round(result["power_W"], 2),
            "efficiency":           round(result["efficiency"], 6),
            "E_rev_V":              round(result["E_rev_V"], 6),
            "V_act_V":              round(result["V_act_V"], 6),
            "V_ohm_V":              round(result["V_ohm_V"], 6),
        }

    def get_info(self) -> dict:
        """Return component metadata and parameter summary."""
        return {
            "component_id":    self.component_id,
            "component_name":  self.component_name,
            "fidelity":        self.fidelity,
            "version":         self.version,
            "inputs": {
                "current_density": {"unit": "A/cm²", "range": [0.0, 2.0]},
                "temperature":     {"unit": "°C",    "range": [40.0, 90.0]},
            },
            "outputs": {
                "cell_voltage_V":      "V",
                "stack_voltage_V":     "V",
                "hydrogen_rate_mol_s": "mol/s",
                "power_W":             "W",
                "efficiency":          "-",
                "E_rev_V":             "V",
                "V_act_V":             "V",
                "V_ohm_V":             "V",
            },
            "active_parameters": self._params,
            "source": (
                "Garcia-Valverde et al. (2012), Int. J. Hydrogen Energy, "
                "37(2), 1927-1938"
            ),
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(model.get_info())
    print(model.predict({"current_density": 1.0, "temperature": 80.0}))
