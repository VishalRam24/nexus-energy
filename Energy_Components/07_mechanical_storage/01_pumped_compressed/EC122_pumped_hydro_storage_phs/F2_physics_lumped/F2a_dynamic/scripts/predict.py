"""
EC122 -- Pumped Hydro Storage (PHS) -- F2a Dynamic
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import PumpedHydroStorage_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for PHS F2a dynamic model."""

    component_id = "EC122"
    component_name = "Pumped Hydro Storage (PHS)"
    fidelity = "F2a -- Dynamic ODE Model with Waterway Inertia"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = PumpedHydroStorage_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation.

        inputs:
            P_electrical_W : float -- power demand [W] (positive=generate)
            mode : str -- 'turbine', 'pump', or 'idle'
            Q0 : float -- initial flow [m3/s] (default 0)
            omega0 : float -- initial speed [rad/s]
            H_up0 : float -- initial upper level [m]
            H_low0 : float -- initial lower level [m]
            dt : float -- time step [s] (default 1.0)
            duration_s : float -- simulation time [s] (default 3600)
        """
        P = inputs.get("P_electrical_W", 200e6)
        mode = inputs.get("mode", "turbine")
        Q0 = inputs.get("Q0", 0.0)
        H_up0 = inputs.get("H_up0", None)
        H_lo0 = inputs.get("H_low0", None)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 3600.0)

        return self._model.simulate(P, mode, Q0, H_up0, H_lo0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "P_electrical_W": {"unit": "W", "range": [-300e6, 300e6]},
                "mode": {"type": "str", "options": ["turbine", "pump", "idle"]},
                "Q0": {"unit": "m3/s", "range": [-150, 150]},
                "omega0": {"unit": "rad/s"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s", "Q": "m3/s", "omega": "rad/s",
                "H_upper": "m", "H_lower": "m", "H_net": "m",
                "P_hydraulic": "W", "P_electrical": "W",
                "efficiency": "-", "SOC": "-", "E_stored": "J",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"P_electrical_W": 200e6, "mode": "turbine", "duration_s": 60.0, "dt": 1.0})
    print(f"Final Q: {r['Q'][-1]:.2f} m3/s, SOC: {r['SOC'][-1]:.4f}")
