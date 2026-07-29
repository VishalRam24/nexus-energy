"""
EC011 -- AEM Electrolyser -- F1b V-I Thermal
ComponentModel wrapper with standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import AEMThermalModel

_PARAMS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "parameters.json"
)


def _load_default_params() -> dict:
    with open(_PARAMS_PATH, "r") as fh:
        return json.load(fh)


class ComponentModel:
    """
    Standardised wrapper for the AEM electrolyser thermal polarization model.

    Usage
    -----
    >>> model = ComponentModel()
    >>> out = model.predict({"current_density": 5000.0, "temperature": 333.15})
    """

    component_id   = "EC011"
    component_name = "Anion Exchange Membrane Electrolyser (AEM)"
    fidelity       = "F1b -- V-I Thermal (T-coupled polarization + thermal balance)"
    version        = "1.0.0"

    def __init__(self, params: dict = None):
        defaults = _load_default_params()
        if params:
            defaults["unit"].update(params)
        self._raw = defaults
        self._physics = AEMThermalModel(defaults)

    def predict(self, inputs: dict) -> dict:
        """
        Predict AEM operating point.

        Parameters
        ----------
        inputs : dict
            current_density : float -- A/m2  (valid: 0 – 20 000)
            temperature     : float -- K     (valid: 303.15 – 363.15); optional if
                                             solve_thermal=True
            T_coolant       : float -- K     (optional, default from params)
            solve_thermal   : bool  -- if True, compute self-consistent steady-state T
                                       (ignores supplied temperature); default False

        Returns
        -------
        dict with cell_voltage_V, stack_voltage_V, power_kW, heat_generation_W,
             ASR_ohm_cm2, hydrogen_rate_mol_s, efficiency_lhv, temperature_K
        """
        j = float(inputs["current_density"])
        if not (0.0 <= j <= 20000.0):
            raise ValueError(f"current_density must be in [0, 20000] A/m2, got {j}")

        solve_T = bool(inputs.get("solve_thermal", False))
        T_cool = float(inputs.get("T_coolant", self._physics.T_cool_default))

        if solve_T:
            T = float(self._physics.steady_state_temperature(j, T_cool))
        else:
            T = float(inputs.get("temperature", self._physics.T_ref))
            if not (303.15 <= T <= 363.15):
                raise ValueError(f"temperature must be in [303.15, 363.15] K, got {T}")

        res = self._physics.evaluate(j, T)

        return {
            "cell_voltage_V":         round(float(res["cell_voltage_V"]), 6),
            "stack_voltage_V":        round(float(res["stack_voltage_V"]), 6),
            "power_kW":               round(float(res["power_kW"]), 6),
            "heat_generation_W":      round(float(res["heat_generation_W"]), 4),
            "ASR_ohm_cm2":            round(float(res["ASR_ohm_cm2"]), 6),
            "hydrogen_rate_mol_s":    round(float(res["hydrogen_rate_mol_s"]), 8),
            "efficiency_lhv":         round(float(res["efficiency_lhv"]), 6),
            "temperature_K":          round(T, 4),
        }

    def get_info(self) -> dict:
        p = self._physics
        return {
            "component_id":   self.component_id,
            "component_name": self.component_name,
            "fidelity":       self.fidelity,
            "version":        self.version,
            "inputs": {
                "current_density": {"unit": "A/m2",  "range": [0.0, 20000.0]},
                "temperature":     {"unit": "K",      "range": [303.15, 363.15], "optional": True},
                "T_coolant":       {"unit": "K",      "range": [293.15, 353.15], "optional": True},
                "solve_thermal":   {"unit": "bool",   "optional": True},
            },
            "outputs": {
                "cell_voltage_V":       "V",
                "stack_voltage_V":      "V",
                "power_kW":             "kW",
                "heat_generation_W":    "W",
                "ASR_ohm_cm2":          "Ohm.cm2",
                "hydrogen_rate_mol_s":  "mol/s",
                "efficiency_lhv":       "-",
                "temperature_K":        "K",
            },
            "thermal_params": {
                "Cp_stack_J_K":  p.Cp_stack,
                "UA_cool_W_K":   p.UA_cool,
                "T_coolant_K":   p.T_cool_default,
            },
            "source": "Vincent & Bessarabov (2018); Henkensmeier et al. (2021); Schalenbach et al. (2016)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(model.get_info())
    print(model.predict({"current_density": 5000.0, "temperature": 333.15}))
    print("Thermal solve:", model.predict({"current_density": 5000.0, "solve_thermal": True}))
