"""
EC002 -- SOFC -- F2a Electrochemical
Standardised predict() / get_info() interface.
"""

import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import SOFC_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")

def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)

class ComponentModel:
    component_id = "EC002"
    component_name = "Solid Oxide Fuel Cell (SOFC)"
    fidelity = "F2a -- Full Electrochemical + Thermal ODE"
    version = "1.0.0"

    def __init__(self, params=None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = SOFC_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        j = inputs.get("current_density", 0.5)
        T0 = inputs.get("T_cell_K", 1073.15)
        fuel = inputs.get("fuel_composition", None)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 600.0)
        return self._model.simulate(j, T0, fuel, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_density": {"unit": "A/cm2", "range": [0, 2.5]},
                "T_cell_K": {"unit": "K", "range": [873, 1273]},
                "fuel_composition": "dict with pH2, pO2, pH2O keys",
                "dt": {"unit": "s"}, "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s", "voltage": "V", "power_density": "W/cm2",
                "efficiency": "-", "temperature": "K", "fuel_utilization": "-",
            },
            "source": self._raw.get("source", ""),
        }

if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"current_density": 0.5, "duration_s": 10, "dt": 1})
    print(f"V_final={r['voltage'][-1]:.4f} V, T_final={r['temperature'][-1]:.2f} K")
