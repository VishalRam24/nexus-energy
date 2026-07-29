"""
EC022 -- LCO Battery (Lithium Cobalt Oxide) -- F2a Thevenin 1-RC ECM
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import LCO_ECM_1RC

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for LCO F2a Thevenin 1-RC ECM."""

    component_id = "EC022"
    component_name = "LCO Battery (Lithium Cobalt Oxide)"
    fidelity = "F2a -- Thevenin 1-RC ECM + Coulomb SOC + Bernardi Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = LCO_ECM_1RC(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic ECM simulation.

        inputs:
            current_A   : float or callable(t)  (I>0 discharge, I<0 charge)
            soc0        : float  initial SOC (default 0.9)
            T_K         : float  initial temperature [K] (default ambient)
            v_rc0       : float  initial RC voltage [V] (default 0.0)
            dt          : float  output step [s] (default 1.0)
            duration_s  : float  total duration [s] (default 600.0)
        """
        I = inputs.get("current_A", 2.6)
        soc0 = inputs.get("soc0", 0.9)
        T0 = inputs.get("T_K", None)
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
                "current_A": {"unit": "A", "range": [-10, 10],
                              "note": "I>0 discharge, I<0 charge"},
                "soc0": {"unit": "-", "range": [0, 1]},
                "T_K": {"unit": "K", "range": [273.15, 318.15]},
                "v_rc0": {"unit": "V"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "soc": "-",
                "voltage": "V",
                "current": "A",
                "power": "W",
                "v_rc": "V",
                "temperature": "K",
                "heat_gen": "W",
                "R0": "Ohm",
                "R1": "Ohm",
                "efficiency": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"current_A": 2.6, "soc0": 0.9, "dt": 5.0, "duration_s": 600.0})
    print(f"Final SOC: {r['soc'][-1]:.4f}, "
          f"Final V: {r['voltage'][-1]:.4f} V, "
          f"Final T: {r['temperature'][-1]:.2f} K, "
          f"Peak Q_gen: {r['heat_gen'].max():.3f} W")
