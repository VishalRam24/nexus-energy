"""
EC021 -- LTO Battery (Lithium Titanate Oxide) -- F2a Thevenin 1-RC ECM + Thermal
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import LTO_ECM_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for LTO F2a Thevenin 1-RC equivalent-circuit model."""

    component_id = "EC021"
    component_name = "LTO Battery (Lithium Titanate Oxide)"
    fidelity = "F2a -- Thevenin 1-RC Equivalent-Circuit Model with Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = LTO_ECM_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic 1-RC + thermal simulation.

        inputs:
            current_A   : float (or callable t->A); positive=discharge, negative=charge
            soc0        : float  (initial SOC 0-1, default 0.9)
            T0          : float  (initial cell temperature [K], default 298.15)
            v_rc0       : float  (initial RC polarization [V], default 0.0)
            dt          : float  (output step [s], default 1.0)
            duration_s  : float  (total duration [s], default 600.0)
        """
        I = inputs.get("current_A", 2.9)
        soc0 = inputs.get("soc0", 0.9)
        T0 = inputs.get("T0", 298.15)
        v_rc0 = inputs.get("v_rc0", 0.0)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 600.0)

        return self._model.simulate(I, soc0, T0, v_rc0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_A": {"unit": "A", "range": [-30.0, 30.0],
                              "note": "positive=discharge, negative=charge"},
                "soc0": {"unit": "-", "range": [0.0, 1.0]},
                "T0": {"unit": "K", "range": [243.15, 333.15]},
                "v_rc0": {"unit": "V"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "soc": "-",
                "voltage": "V",
                "v_rc": "V",
                "current": "A",
                "temperature": "K",
                "power": "W",
                "heat_gen": "W",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    # 1C discharge (2.9 A) for 600 s from full charge
    r = m.predict({"current_A": 2.9, "soc0": 0.9, "duration_s": 600.0, "dt": 5.0})
    print(
        f"Final SOC: {r['soc'][-1]:.4f}, "
        f"Final voltage: {r['voltage'][-1]:.4f} V, "
        f"Final T: {r['temperature'][-1]:.3f} K, "
        f"V_RC settled: {r['v_rc'][-1]:.5f} V"
    )
