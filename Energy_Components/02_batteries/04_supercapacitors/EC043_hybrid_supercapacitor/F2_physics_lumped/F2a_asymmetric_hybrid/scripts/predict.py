"""
EC043 -- Hybrid Supercapacitor (Lithium-Ion Capacitor) -- F2a Asymmetric Hybrid
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import HybridSupercapacitorF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for hybrid-supercapacitor F2a asymmetric model."""

    component_id = "EC043"
    component_name = "Hybrid Supercapacitor (Lithium-Ion Capacitor)"
    fidelity = "F2a -- Asymmetric Hybrid Physics-Lumped Model with Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = HybridSupercapacitorF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation.

        inputs:
            current_A   : float (or callable)   load current; +discharge, -charge
            q0_C        : float                 initial charge [C]  (default 0.5*Q_max)
            T0_K        : float                 initial temperature [K] (default 298.15)
            dt          : float                 output step [s]     (default 0.5)
            duration_s  : float                 total time [s]      (default 60.0)
        """
        Q_max = self._model.Q_max
        I = inputs.get("current_A", 50.0)
        q0 = inputs.get("q0_C", 0.5 * Q_max)
        T0 = inputs.get("T0_K", 298.15)
        dt = inputs.get("dt", 0.5)
        dur = inputs.get("duration_s", 60.0)

        return self._model.simulate(I, q0, T0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_A": {"unit": "A", "range": [-600, 600]},
                "q0_C": {"unit": "C", "range": [0, self._model.Q_max]},
                "T0_K": {"unit": "K", "range": [233.15, 343.15]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "charge": "C",
                "soc": "-",
                "v_oc": "V",
                "v_terminal": "V",
                "power": "W",
                "efficiency": "-",
                "temperature": "K",
                "energy_J": "J",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"current_A": 100.0, "duration_s": 20.0, "dt": 1.0})
    print(
        f"V_term start={r['v_terminal'][0]:.3f} V -> end={r['v_terminal'][-1]:.3f} V, "
        f"SOC {r['soc'][0]:.2f}->{r['soc'][-1]:.2f}, T_final={r['temperature'][-1]:.3f} K"
    )
