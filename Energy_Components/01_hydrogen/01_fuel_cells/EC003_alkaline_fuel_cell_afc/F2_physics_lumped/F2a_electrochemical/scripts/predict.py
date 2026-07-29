"""
EC003 -- Alkaline Fuel Cell (AFC) -- F2a Electrochemical
Standardised predict() / get_info() interface (mirrors EC001 ComponentModel).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import AFC_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for AFC F2a full electrochemical model."""

    component_id = "EC003"
    component_name = "Alkaline Fuel Cell (AFC)"
    fidelity = "F2a -- Full Electrochemical Model with KOH Conductivity and Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = AFC_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation.

        inputs:
            current_density_A_cm2 : float (or callable for time-varying)
            T_cell_K : float   (initial temperature, default 343.15)
            P_h2_atm : float   (default 1.0)
            P_o2_atm : float   (default 0.21)
            c_KOH : float      (KOH concentration mol/L, default from params)
            dt : float         (default 0.5)
            duration_s : float (default 60.0)
        """
        j = inputs.get("current_density_A_cm2", 0.4)
        T0 = inputs.get("T_cell_K", 343.15)
        P_h2 = inputs.get("P_h2_atm", 1.0)
        P_o2 = inputs.get("P_o2_atm", 0.21)
        c_KOH = inputs.get("c_KOH", None)
        dt = inputs.get("dt", 0.5)
        dur = inputs.get("duration_s", 60.0)

        return self._model.simulate(j, T0, P_h2, P_o2, dt, dur, c_KOH=c_KOH)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_density_A_cm2": {"unit": "A/cm2", "range": [0, 1.0]},
                "T_cell_K": {"unit": "K", "range": [313.15, 363.15]},
                "P_h2_atm": {"unit": "atm", "range": [0.5, 4.0]},
                "P_o2_atm": {"unit": "atm", "range": [0.1, 1.0]},
                "c_KOH": {"unit": "mol/L", "range": [1.0, 12.0]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "voltage": "V",
                "power_density": "W/cm2",
                "efficiency": "-",
                "temperature": "K",
                "koh_conductivity": "S/cm",
                "overpotentials": "dict of V arrays",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"current_density_A_cm2": 0.4, "duration_s": 10.0, "dt": 1.0})
    print(f"Final voltage: {r['voltage'][-1]:.4f} V, "
          f"Final T: {r['temperature'][-1]:.2f} K, "
          f"KOH kappa: {r['koh_conductivity'][-1]:.4f} S/cm")
