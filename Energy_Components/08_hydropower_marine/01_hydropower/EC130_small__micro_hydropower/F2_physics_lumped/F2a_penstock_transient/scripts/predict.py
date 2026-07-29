"""
EC130 -- Small/Micro Hydropower -- F2a Penstock Transient
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import MicroHydroF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC130 micro-hydro F2a penstock-transient model."""

    component_id = "EC130"
    component_name = "Small/Micro Hydropower"
    fidelity = "F2a -- Physics-Lumped Penstock Transient (rigid water-column ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = MicroHydroF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic transient simulation.

        inputs:
            gate_command : float in [0,1] (or callable t->[0,1]) gate opening
            Q_in_m3s     : float inflow to forebay (default: gate*Q_design)
            v0           : float initial penstock velocity [m/s] (default: eq.)
            z0           : float initial surge level [m] (default 0)
            g0           : float initial gate position (default: gate_command(0))
            dt           : float output step [s] (default 0.1)
            duration_s   : float total time [s] (default 120.0)
        """
        gate = inputs.get("gate_command", 1.0)
        Q_in = inputs.get("Q_in_m3s", None)
        v0 = inputs.get("v0", None)
        z0 = inputs.get("z0", 0.0)
        g0 = inputs.get("g0", None)
        dt = inputs.get("dt", 0.1)
        dur = inputs.get("duration_s", 120.0)

        return self._model.simulate(gate, Q_in, v0, z0, g0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "gate_command": {"unit": "-", "range": [0.0, 1.0]},
                "Q_in_m3s": {"unit": "m3/s", "range": [0.0, 1.65]},
                "v0": {"unit": "m/s"},
                "z0": {"unit": "m"},
                "g0": {"unit": "-"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "velocity": "m/s",
                "flow": "m3/s",
                "head_net": "m",
                "head_loss": "m",
                "surge_level": "m",
                "gate": "-",
                "power_el": "kW",
                "power_hyd": "kW",
                "efficiency": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    info = m.get_info()
    print(f"{info['component_id']} {info['component_name']} -- {info['fidelity']}")
    r = m.predict({"gate_command": 1.0, "duration_s": 60.0, "dt": 0.5})
    print(f"Final flow:  {r['flow'][-1]:.3f} m3/s")
    print(f"Net head:    {r['head_net'][-1]:.2f} m  (loss {r['head_loss'][-1]:.2f} m)")
    print(f"Efficiency:  {r['efficiency'][-1]:.3f}")
    print(f"Power (el):  {r['power_el'][-1]:.1f} kW")
