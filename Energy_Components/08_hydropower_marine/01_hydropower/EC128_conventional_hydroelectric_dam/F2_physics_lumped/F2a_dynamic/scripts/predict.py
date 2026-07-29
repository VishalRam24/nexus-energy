"""
EC128 -- Conventional Hydroelectric Dam -- F2a Dynamic
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import ConventionalHydroDam_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for Hydro Dam F2a dynamic model."""

    component_id = "EC128"
    component_name = "Conventional Hydroelectric Dam"
    fidelity = "F2a -- Dynamic ODE Model with Penstock + Turbine-Governor"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = ConventionalHydroDam_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation.

        inputs:
            G_ref : float -- gate opening reference (0-1)
            Q_inflow : float -- natural inflow [m3/s]
            H0 : float -- initial head [m]
            dt : float -- time step [s]
            duration_s : float -- simulation time [s]
        """
        G_ref = inputs.get("G_ref", 0.5)
        Q_in = inputs.get("Q_inflow", None)
        H0 = inputs.get("H0", None)
        Q0 = inputs.get("Q0", None)
        G0 = inputs.get("G0", None)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 3600.0)

        return self._model.simulate(G_ref, Q_in, None, H0, Q0, G0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "G_ref": {"unit": "-", "range": [0, 1]},
                "Q_inflow": {"unit": "m3/s"},
                "H0": {"unit": "m"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s", "H_reservoir": "m", "Q_penstock": "m3/s",
                "Q_turbine": "m3/s", "G_gate": "-",
                "P_output": "W", "efficiency": "-", "Q_inflow": "m3/s",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"G_ref": 0.8, "duration_s": 60.0, "dt": 1.0})
    print(f"Final P: {r['P_output'][-1]/1e6:.2f} MW, H: {r['H_reservoir'][-1]:.2f} m")
