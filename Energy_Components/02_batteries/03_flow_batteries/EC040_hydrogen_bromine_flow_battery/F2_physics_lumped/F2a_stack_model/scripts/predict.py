"""
EC040 -- Hydrogen-Bromine Flow Battery (HBrFB) -- F2a Physics-Lumped Stack Model
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import HydrogenBromineFlowF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for HBrFB F2a physics-lumped stack model."""

    component_id = "EC040"
    component_name = "Hydrogen-Bromine Flow Battery (HBrFB)"
    fidelity = "F2a -- Physics-Lumped Stack (kinetics + SOC ODE + Br crossover + thermal ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = HydrogenBromineFlowF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation of the H2/Br2 stack.

        inputs:
            current_A   : float (or callable(t))  stack current; >0 discharge, <0 charge
            soc0        : float  initial state of charge (default 0.5)
            T0          : float  initial temperature [K] (default 298.15)
            dt          : float  output time step [s] (default 10.0)
            duration_s  : float  total duration [s] (default 3600.0)
        """
        I = inputs.get("current_A", 50.0)
        soc0 = inputs.get("soc0", 0.5)
        T0 = inputs.get("T0", 298.15)
        dt = inputs.get("dt", 10.0)
        dur = inputs.get("duration_s", 3600.0)
        return self._model.simulate(I, soc0, T0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_A": {"unit": "A", "range": [-300, 300], "note": ">0 discharge, <0 charge"},
                "soc0": {"unit": "-", "range": [0.02, 0.98]},
                "T0": {"unit": "K", "range": [293.15, 333.15]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "soc": "-",
                "temperature": "K",
                "cell_voltage": "V",
                "stack_voltage": "V",
                "power_W": "W",
                "E_nernst": "V",
                "overpotentials": "dict of V arrays",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"current_A": 50.0, "soc0": 0.7, "duration_s": 600.0, "dt": 60.0})
    print(f"Discharge @50A: V_stack={r['stack_voltage'][0]:.2f} V -> {r['stack_voltage'][-1]:.2f} V, "
          f"SOC {r['soc'][0]:.3f} -> {r['soc'][-1]:.3f}, "
          f"T {r['temperature'][0]:.2f} -> {r['temperature'][-1]:.2f} K")
    rt = m._model.round_trip_efficiency(50.0, soc=0.5, T=298.15)
    print(f"Round-trip efficiency @50A: {rt*100:.1f} %")
