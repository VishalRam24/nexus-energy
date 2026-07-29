"""
EC001 — PEM Fuel Cell (PEMFC) — F1a Polarization Curve
ComponentModel wrapper with standardised predict() / get_info() interface.
"""

import json
import os
from model import PEMFuelCellModel

_PARAMS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "parameters.json"
)


def _load_default_params() -> dict:
    with open(_PARAMS_PATH, "r") as fh:
        return json.load(fh)["default_parameters"]


class ComponentModel:
    """
    Standardised wrapper for the PEM Fuel Cell polarization-curve model.

    Usage
    -----
    >>> model = ComponentModel()
    >>> out = model.predict({"current_density": 0.6, "temperature": 70.0})
    >>> print(out)
    """

    component_id   = "EC001"
    component_name = "PEM Fuel Cell (PEMFC)"
    fidelity       = "F1a — Polarization Curve (Amphlett semi-empirical)"
    version        = "1.0.0"

    def __init__(self, params: dict = None):
        defaults = _load_default_params()
        if params:
            defaults.update(params)
        self._params  = defaults
        self._physics = PEMFuelCellModel(defaults)

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def predict(self, inputs: dict) -> dict:
        """
        Predict fuel cell operating point.

        Parameters
        ----------
        inputs : dict
            current_density : float  — A/cm²  (0 – j_L exclusive)
            temperature     : float  — °C     (50 – 90), optional

        Returns
        -------
        dict
            cell_voltage_V, stack_voltage_V, power_density_W_cm2,
            stack_power_W, efficiency, E_Nernst_V, V_act_V, V_ohm_V, V_conc_V
        """
        j   = float(inputs["current_density"])
        T_C = float(inputs.get("temperature", self._params["T"] - 273.15))

        j_L = self._physics.j_L
        if not (0.0 <= j < j_L):
            raise ValueError(
                f"current_density must be in [0, {j_L}) A/cm², got {j}"
            )
        if not (50.0 <= T_C <= 90.0):
            raise ValueError(f"temperature must be in [50, 90] °C, got {T_C}")

        result = self._physics.evaluate(j, T_C)

        return {
            "cell_voltage_V":        round(result["cell_voltage_V"],        6),
            "stack_voltage_V":       round(result["stack_voltage_V"],       4),
            "power_density_W_cm2":   round(result["power_density_W_cm2"],   6),
            "stack_power_W":         round(result["stack_power_W"],         2),
            "efficiency":            round(result["efficiency"],             6),
            "E_Nernst_V":            round(result["E_Nernst_V"],            6),
            "V_act_V":               round(result["V_act_V"],               6),
            "V_ohm_V":               round(result["V_ohm_V"],               6),
            "V_conc_V":              round(result["V_conc_V"],              6),
        }

    def get_info(self) -> dict:
        """Return component metadata and parameter summary."""
        return {
            "component_id":   self.component_id,
            "component_name": self.component_name,
            "fidelity":       self.fidelity,
            "version":        self.version,
            "inputs": {
                "current_density": {
                    "unit": "A/cm²",
                    "range": [0.0, self._physics.j_L]
                },
                "temperature": {"unit": "°C", "range": [50.0, 90.0]},
            },
            "outputs": {
                "cell_voltage_V":      "V",
                "stack_voltage_V":     "V",
                "power_density_W_cm2": "W/cm²",
                "stack_power_W":       "W",
                "efficiency":          "-",
                "E_Nernst_V":          "V",
                "V_act_V":             "V",
                "V_ohm_V":             "V",
                "V_conc_V":            "V",
            },
            "active_parameters": self._params,
            "source": (
                "Amphlett et al. (1995), J. Electrochem. Soc., 142(1), 1-8"
            ),
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(model.get_info())
    print(model.predict({"current_density": 0.6, "temperature": 70.0}))
