"""
EC159 -- Buck-Boost Converter (Inverting) -- F2a State-Space Averaged Model
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import BuckBoostConverterF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the buck-boost F2a averaged state-space model."""

    component_id = "EC159"
    component_name = "Buck-Boost Converter (Inverting)"
    fidelity = "F2a -- State-Space Averaged Model (lumped ODE, CCM)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = BuckBoostConverterF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a transient simulation of the averaged converter, plus its
        steady-state operating point.

        inputs:
            duty        : float (or callable)  -- duty cycle in [0,1]
            v_in        : float (default nominal)  -- input voltage [V]
            R_load      : float (default nominal)  -- load resistance [Ohm]
            dt          : float (default 1e-6) -- output step [s]
            duration_s  : float (default 2e-3) -- total time [s]
            iL0, vC0    : float initial states (default 0)

        Returns time-series arrays + a 'steady_state' dict.
        """
        duty = inputs.get("duty", 0.5)
        v_in = inputs.get("v_in", self._model.Vin_nom)
        R_load = inputs.get("R_load", self._model.R_load_nom)
        dt = inputs.get("dt", 1.0e-6)
        dur = inputs.get("duration_s", 2.0e-3)
        iL0 = inputs.get("iL0", 0.0)
        vC0 = inputs.get("vC0", 0.0)

        ts = self._model.simulate(duty, v_in, R_load, dt, dur, iL0, vC0)
        d_ss = duty if not callable(duty) else duty(dur)
        ts["steady_state"] = self._model.steady_state(d_ss, v_in, R_load)
        return ts

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "duty": {"unit": "-", "range": [0.05, 0.95]},
                "v_in": {"unit": "V", "range": [5.0, 60.0]},
                "R_load": {"unit": "Ohm", "range": [0.5, 100.0]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
                "iL0": {"unit": "A"},
                "vC0": {"unit": "V"},
            },
            "outputs": {
                "t": "s",
                "iL": "A (inductor current)",
                "vC": "V (capacitor voltage magnitude)",
                "vout": "V (inverting output = -vC)",
                "p_in": "W",
                "p_out": "W",
                "efficiency": "-",
                "steady_state": "dict (iL, vC, vout, gain, efficiency)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"duty": 0.5, "v_in": 24.0, "R_load": 4.0,
                   "dt": 2.0e-6, "duration_s": 4.0e-3})
    ss = r["steady_state"]
    print(f"Settled Vout: {r['vout'][-1]:.3f} V  (steady-state Vout: {ss['vout']:.3f} V)")
    print(f"Inductor current: {r['iL'][-1]:.3f} A,  efficiency: {ss['efficiency']*100:.2f}%")
    print(f"Ideal gain at d=0.5: {m._model.ideal_gain(0.5):.3f}  (expect -1.0)")
