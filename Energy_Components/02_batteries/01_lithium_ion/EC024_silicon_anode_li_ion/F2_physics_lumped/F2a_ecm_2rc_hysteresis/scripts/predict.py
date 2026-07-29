"""
EC024 -- Silicon-Anode Li-ion Battery (Si/NMC) -- F2a Thevenin ECM
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import SiAnodeECM_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC024 Si-anode Li-ion F2a Thevenin ECM."""

    component_id = "EC024"
    component_name = "Silicon-Anode Li-ion Battery (Si/NMC)"
    fidelity = "F2a -- Thevenin ECM (1/2-RC) + Coulomb SOC + Si hysteresis + thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = SiAnodeECM_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic ECM simulation.

        inputs:
            current_A : float (or callable(t))  I>0 discharge, I<0 charge
            soc0 : float        initial SOC (default 0.9)
            T_K : float         initial temperature (default 298.15)
            dt : float          output step [s] (default 1.0)
            duration_s : float  total duration [s] (default 600.0)
            h0 : float or None  initial hysteresis state (default auto)
        """
        I = inputs.get("current_A", 3.5)
        soc0 = inputs.get("soc0", 0.9)
        T0 = inputs.get("T_K", 298.15)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 600.0)
        h0 = inputs.get("h0", None)
        return self._model.simulate(I, soc0, T0, dt, dur, h0=h0)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_A": {"unit": "A", "range": [-17.5, 17.5],
                              "note": "I>0 discharge, I<0 charge"},
                "soc0": {"unit": "-", "range": [0.0, 1.0]},
                "T_K": {"unit": "K", "range": [253.15, 333.15]},
                "dt": {"unit": "s", "range": [0.01, 10.0]},
                "duration_s": {"unit": "s", "range": [1.0, 36000.0]},
                "h0": {"unit": "-", "range": [-1.0, 1.0]},
            },
            "outputs": {
                "t": "s",
                "soc": "-",
                "voltage": "V",
                "ocv": "V",
                "current": "A",
                "power": "W",
                "v_rc1": "V",
                "v_rc2": "V",
                "hysteresis": "-",
                "temperature": "K",
                "swelling_strain": "-",
                "efficiency": "-",
                "components": "dict of arrays",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"current_A": 3.5, "soc0": 0.9, "duration_s": 600.0, "dt": 5.0})
    print(f"Final SOC: {r['soc'][-1]:.4f}, "
          f"Final V: {r['voltage'][-1]:.4f} V, "
          f"Final T: {r['temperature'][-1]:.2f} K, "
          f"swelling: {r['swelling_strain'][-1]*100:.1f}%")
