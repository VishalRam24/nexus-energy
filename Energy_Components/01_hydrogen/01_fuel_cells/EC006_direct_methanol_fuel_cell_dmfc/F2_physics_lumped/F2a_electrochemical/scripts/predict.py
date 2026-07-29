"""
EC006 -- Direct Methanol Fuel Cell (DMFC) -- F2a Electrochemical
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import DMFC_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for DMFC F2a full electrochemical model."""

    component_id = "EC006"
    component_name = "Direct Methanol Fuel Cell (DMFC)"
    fidelity = "F2a -- Full Electrochemical Model with Methanol Crossover + Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = DMFC_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation.

        inputs:
            current_density_A_cm2 : float (or callable for time-varying)
            T_cell_K       : float  (initial temperature, default 343.15)
            c_MeOH_molar   : float  (methanol feed concentration, default 1.0)
            dt             : float  (default 0.5)
            duration_s     : float  (default 60.0)
        """
        j = inputs.get("current_density_A_cm2", 0.2)
        T0 = inputs.get("T_cell_K", 343.15)
        c = inputs.get("c_MeOH_molar", 1.0)
        dt = inputs.get("dt", 0.5)
        dur = inputs.get("duration_s", 60.0)

        return self._model.simulate(j, T0, c, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_density_A_cm2": {"unit": "A/cm2", "range": [0, 0.4]},
                "T_cell_K": {"unit": "K", "range": [313.15, 383.15]},
                "c_MeOH_molar": {"unit": "mol/L", "range": [0.25, 2.0]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "voltage": "V",
                "power_density": "W/cm2",
                "efficiency": "- (overall = voltage_eff * fuel_eff)",
                "fuel_efficiency": "-",
                "temperature": "K",
                "crossover_current": "A/cm2",
                "overpotentials": "dict of V arrays",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"current_density_A_cm2": 0.15, "duration_s": 10.0, "dt": 1.0})
    print(
        f"Final voltage: {r['voltage'][-1]:.4f} V, "
        f"Final T: {r['temperature'][-1]:.2f} K, "
        f"fuel_eff: {r['fuel_efficiency'][-1]:.3f}, "
        f"j_cross: {r['crossover_current'][-1]:.4f} A/cm2"
    )
