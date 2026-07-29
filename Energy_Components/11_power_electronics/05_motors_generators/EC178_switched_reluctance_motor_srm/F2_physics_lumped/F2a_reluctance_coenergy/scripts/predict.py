"""
EC178 -- Switched Reluctance Motor (SRM) -- F2a Reluctance / Co-energy Model
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import SRM_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the SRM F2a physics-lumped reluctance model."""

    component_id = "EC178"
    component_name = "Switched Reluctance Motor (SRM)"
    fidelity = "F2a -- Physics-Lumped Reluctance / Co-energy Model"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = SRM_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a coupled electromechanical simulation.

        inputs:
            V_dc : float        DC-link voltage [V] (default from parameters)
            T_load : float      Load torque [N.m] (default 2.0)
            omega0 : float      Initial speed [rad/s] (default rated)
            theta0 : float      Initial rotor angle [rad] (default 0)
            dt : float          Output time step [s] (default 2e-5)
            duration_s : float  Duration [s] (default 0.05)
        """
        V_dc = inputs.get("V_dc", None)
        T_load = inputs.get("T_load", 2.0)
        omega0 = inputs.get("omega0", None)
        theta0 = inputs.get("theta0", 0.0)
        dt = inputs.get("dt", 2e-5)
        dur = inputs.get("duration_s", 0.05)

        return self._model.simulate(
            V_dc=V_dc, T_load=T_load, omega0=omega0,
            theta0=theta0, dt=dt, duration_s=dur,
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "V_dc": {"unit": "V", "range": [50, 600]},
                "T_load": {"unit": "N.m", "range": [0, 60]},
                "omega0": {"unit": "rad/s"},
                "theta0": {"unit": "rad"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "theta": "rad",
                "omega": "rad/s",
                "speed_rpm": "rpm",
                "torque": "N.m",
                "phase_currents": "A (Nph x N)",
                "T_avg": "N.m",
                "torque_ripple": "-",
                "efficiency": "-",
                "W_elec_J / W_mech_J / W_copper_J": "J",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"T_load": 2.0, "duration_s": 0.04, "dt": 2e-5})
    print(
        f"T_avg = {r['T_avg']:.3f} N.m | ripple = {r['torque_ripple']:.3f} | "
        f"eff = {r['efficiency']:.3f} | speed = {r['speed_rpm'][-1]:.0f} rpm"
    )
