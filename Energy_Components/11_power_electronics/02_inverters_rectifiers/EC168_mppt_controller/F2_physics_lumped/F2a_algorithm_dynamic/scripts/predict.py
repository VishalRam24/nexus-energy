"""
EC168 -- MPPT Controller -- F2a Algorithm Dynamic
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import MPPTController_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC168 MPPT Controller F2a model."""

    component_id = "EC168"
    component_name = "MPPT Controller"
    fidelity = "F2a -- P&O Algorithm + Buck Converter Dynamics"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = MPPTController_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation.

        inputs:
            irradiance : float or list   Solar irradiance [W/m2] (default 1000)
            T_cell : float               Cell temperature [K] (default 298.15)
            dt : float                   Output time step [s] (default 0.001)
            duration_s : float           Simulation duration [s] (default 1.0)
        """
        G = inputs.get("irradiance", 1000.0)
        T = inputs.get("T_cell", 298.15)
        dt = inputs.get("dt", 0.001)
        dur = inputs.get("duration_s", 1.0)

        result = self._model.simulate(G, T, dt, dur)
        return result

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "irradiance": {"unit": "W/m2", "range": [0, 1200]},
                "T_cell": {"unit": "K", "range": [273.15, 348.15]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "V_pv": "V",
                "I_pv": "A",
                "P_pv": "W",
                "V_ref": "V",
                "duty_cycle": "-",
                "I_L": "A",
                "V_out": "V",
                "P_out": "W",
                "tracking_efficiency": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"irradiance": 1000.0, "dt": 0.001, "duration_s": 0.5})
    print(f"Final P_pv: {r['P_pv'][-1]:.2f} W, "
          f"V_ref: {r['V_ref'][-1]:.2f} V, "
          f"eta_track: {r['tracking_efficiency'][-1]:.4f}")
