"""
EC005 -- Molten Carbonate Fuel Cell (MCFC) -- F2a Electrochemical
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import MCFC_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for MCFC F2a full electrochemical model."""

    component_id = "EC005"
    component_name = "Molten Carbonate Fuel Cell (MCFC)"
    fidelity = "F2a -- Full Electrochemical Model with Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = MCFC_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation.

        inputs:
            current_density_A_cm2 : float (or callable for time-varying)
            T_cell_K     : float  (initial temperature, default 923.15)
            pH2          : float  (default 0.7)
            pO2          : float  (default 0.15)
            pH2O         : float  (default 0.20)
            pCO2_cathode : float  (default 0.15)
            pCO2_anode   : float  (default 0.10)
            dt           : float  (default 1.0)
            duration_s   : float  (default 600.0)
        """
        j = inputs.get("current_density_A_cm2", 0.3)
        T0 = inputs.get("T_cell_K", 923.15)
        pH2 = inputs.get("pH2", 0.7)
        pO2 = inputs.get("pO2", 0.15)
        pH2O = inputs.get("pH2O", 0.20)
        pCO2_cat = inputs.get("pCO2_cathode", 0.15)
        pCO2_an = inputs.get("pCO2_anode", 0.10)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 600.0)

        return self._model.simulate(j, T0, pH2, pO2, pH2O, pCO2_cat, pCO2_an, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_density_A_cm2": {"unit": "A/cm2", "range": [0, 0.6]},
                "T_cell_K": {"unit": "K", "range": [823.15, 973.15]},
                "pH2": {"unit": "atm", "range": [0.1, 1.0]},
                "pO2": {"unit": "atm", "range": [0.05, 0.5]},
                "pH2O": {"unit": "atm", "range": [0.05, 0.5]},
                "pCO2_cathode": {"unit": "atm", "range": [0.05, 0.5]},
                "pCO2_anode": {"unit": "atm", "range": [0.05, 0.5]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "voltage": "V",
                "power_density": "W/cm2",
                "efficiency": "-",
                "temperature": "K",
                "overpotentials": "dict of V arrays",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"current_density_A_cm2": 0.3, "duration_s": 300.0, "dt": 5.0})
    print(f"Final voltage: {r['voltage'][-1]:.4f} V, "
          f"Final T: {r['temperature'][-1]:.2f} K, "
          f"Power density: {r['power_density'][-1]:.4f} W/cm2")
