"""
EC030 -- Nickel-Cadmium Battery (NiCd) -- F2a Thevenin ECM
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import NiCdTheveninF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for NiCd F2a Thevenin equivalent-circuit model."""

    component_id = "EC030"
    component_name = "Nickel-Cadmium Battery (NiCd)"
    fidelity = "F2a -- Thevenin ECM (1-RC / 2-RC) with Coulomb-counted SOC and Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None, n_rc: int = 2):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = NiCdTheveninF2a(self._raw, n_rc=n_rc)

    def predict(self, inputs: dict) -> dict:
        """
        Run a dynamic ECM + thermal simulation.

        inputs:
            current_A   : float or callable(t)  (I>0 discharge, I<0 charge)
            soc0        : float  initial SOC (default 1.0)
            T0          : float  initial temperature K (default ambient)
            dt          : float  output step s (default 1.0)
            duration_s  : float  total time s (default 600.0)
        """
        current = inputs.get("current_A", 5.0)
        soc0 = inputs.get("soc0", 1.0)
        T0 = inputs.get("T0", None)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 600.0)
        return self._model.simulate(current, soc0=soc0, T0=T0, dt=dt, duration_s=dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_A": {"unit": "A", "range": [-50.0, 50.0],
                              "note": "I>0 discharge, I<0 charge; float or callable(t)"},
                "soc0": {"unit": "-", "range": [0.0, 1.0]},
                "T0": {"unit": "K", "range": [243.15, 333.15]},
                "dt": {"unit": "s", "range": [0.01, 5.0]},
                "duration_s": {"unit": "s", "range": [1.0, 36000.0]},
            },
            "outputs": {
                "t": "s",
                "soc": "-",
                "voltage": "V",
                "current": "A",
                "power": "W",
                "efficiency": "-",
                "temperature": "K",
                "ocv": "V",
                "V_rc1": "V",
                "V_rc2": "V",
                "heat_W": "W",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"current_A": 10.0, "soc0": 1.0, "dt": 1.0, "duration_s": 600.0})
    print(f"After 600 s at 10 A (1C) discharge:")
    print(f"  SOC: {r['soc'][0]:.3f} -> {r['soc'][-1]:.3f}")
    print(f"  V_terminal: {r['voltage'][0]:.4f} -> {r['voltage'][-1]:.4f} V")
    print(f"  T: {r['temperature'][0]:.2f} -> {r['temperature'][-1]:.2f} K")
    print(f"  mean efficiency: {r['efficiency'].mean():.4f}")
