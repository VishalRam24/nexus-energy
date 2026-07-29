"""
EC023 -- LMO Battery (Lithium Manganese Oxide) -- F2a Thevenin ECM
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import LMO_Thevenin_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the LMO F2a Thevenin equivalent-circuit model."""

    component_id = "EC023"
    component_name = "LMO Battery (Lithium Manganese Oxide)"
    fidelity = "F2a -- Thevenin ECM (1-RC/2-RC) with Coulomb SOC + Bernardi thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = LMO_Thevenin_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a dynamic ECM + thermal simulation.

        inputs:
            current_A   : float (or callable t->A); +discharge / -charge. default 3.0
            soc0        : float initial SOC in [0,1].  default 0.8
            T0          : float initial temperature [K]. default 298.15
            dt          : float output time step [s].    default 1.0
            duration_s  : float total duration [s].      default 600.0
        """
        I = inputs.get("current_A", 3.0)
        soc0 = inputs.get("soc0", 0.8)
        T0 = inputs.get("T0", 298.15)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 600.0)
        return self._model.simulate(I, soc0, T0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_A": {"unit": "A", "range": [-15.0, 15.0],
                              "note": "+discharge / -charge"},
                "soc0": {"unit": "-", "range": [0.0, 1.0]},
                "T0": {"unit": "K", "range": [253.15, 333.15]},
                "dt": {"unit": "s", "range": [0.01, 10.0]},
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
                "heat_generation": "W",
                "v_rc1": "V",
                "v_rc2": "V",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"current_A": 3.0, "soc0": 0.9, "duration_s": 600.0, "dt": 5.0})
    print(f"Final SOC: {r['soc'][-1]:.4f}  "
          f"Final V: {r['voltage'][-1]:.4f} V  "
          f"Final T: {r['temperature'][-1]:.2f} K")
