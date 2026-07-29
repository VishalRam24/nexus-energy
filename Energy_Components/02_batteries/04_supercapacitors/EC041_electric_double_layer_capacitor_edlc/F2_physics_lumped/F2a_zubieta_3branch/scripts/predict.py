"""
EC041 -- EDLC Supercapacitor -- F2a Zubieta 3-Branch ECM
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import EDLC_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EDLC F2a Zubieta 3-branch ECM with thermal ODE."""

    component_id = "EC041"
    component_name = "Electric Double-Layer Capacitor (EDLC) Supercapacitor"
    fidelity = "F2a -- Zubieta 3-Branch ECM with C(V) + Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = EDLC_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation of the EDLC.

        inputs:
            current_A : float or callable(t)  terminal current [A] (>0 charge)
            v0_V : float        initial branch voltage (default 0.0)
            T0_K : float        initial temperature [K] (default ambient)
            dt : float          output time step [s] (default 0.1)
            duration_s : float  total duration [s] (default 60.0)
        """
        I = inputs.get("current_A", 100.0)
        v0 = inputs.get("v0_V", 0.0)
        T0 = inputs.get("T0_K", None)
        dt = inputs.get("dt", 0.1)
        dur = inputs.get("duration_s", 60.0)
        return self._model.simulate(I, v0, T0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_A": {"unit": "A", "range": [-400, 400], "note": ">0 charge, <0 discharge"},
                "v0_V": {"unit": "V", "range": [0, 2.7]},
                "T0_K": {"unit": "K", "range": [233.15, 338.15]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "v_terminal": "V",
                "v1": "V (immediate branch)",
                "v2": "V (delayed branch)",
                "v3": "V (long-term branch)",
                "current": "A",
                "energy_J": "J (stored)",
                "power_W": "W (terminal, +=discharge)",
                "temperature": "K",
                "esr_Ohm": "Ohm",
                "heat_W": "W (Joule dissipation)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"current_A": 100.0, "duration_s": 30.0, "dt": 1.0})
    print(f"Final terminal V: {r['v_terminal'][-1]:.4f} V, "
          f"Stored E: {r['energy_J'][-1]:.1f} J, "
          f"Final T: {r['temperature'][-1]:.3f} K")
