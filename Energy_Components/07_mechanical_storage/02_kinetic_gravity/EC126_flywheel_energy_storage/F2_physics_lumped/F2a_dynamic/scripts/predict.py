"""
EC126 -- Flywheel Energy Storage -- F2a Dynamic
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import FlywheelStorage_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for Flywheel F2a dynamic model."""

    component_id = "EC126"
    component_name = "Flywheel Energy Storage"
    fidelity = "F2a -- Dynamic Rotational ODE Model"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = FlywheelStorage_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation.

        inputs:
            P_command_W : float -- power command [W] (positive=charge, negative=discharge)
            omega0 : float -- initial angular speed [rad/s]
            dt : float -- time step [s]
            duration_s : float -- simulation time [s]
        """
        P = inputs.get("P_command_W", -100000.0)
        omega0 = inputs.get("omega0", None)
        dt = inputs.get("dt", 0.1)
        dur = inputs.get("duration_s", 3600.0)

        return self._model.simulate(P, omega0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "P_command_W": {"unit": "W", "range": [-250000, 250000]},
                "omega0": {"unit": "rad/s"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s", "omega": "rad/s", "E_stored": "J",
                "SOC": "-", "P_command": "W", "P_loss": "W",
                "efficiency": "-", "T_friction": "N.m",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"P_command_W": -100000, "duration_s": 60.0, "dt": 1.0})
    print(f"Final omega: {r['omega'][-1]:.1f} rad/s, SOC: {r['SOC'][-1]:.4f}")
