"""
EC035 -- Sodium-Sulfur (NaS) Battery -- F2a Thevenin 1-RC ECM
Standardised predict() / get_info() interface (mirrors EC001 F2a template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import NaSBatteryF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for NaS F2a Thevenin 1-RC ECM with thermal ODE."""

    component_id = "EC035"
    component_name = "Sodium-Sulfur (NaS) Battery"
    fidelity = "F2a -- Thevenin 1-RC ECM + Coulomb SOC + Arrhenius beta-alumina R + thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = NaSBatteryF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a dynamic NaS cell simulation.

        inputs:
            current_A   : float (or callable t->A)  >0 discharge, <0 charge (default 20.0)
            soc0        : float  initial SOC 0-1     (default 0.9)
            T0_K        : float  initial temperature (default 593.15 = 320 degC)
            dt          : float  output step [s]     (default 5.0)
            duration_s  : float  total duration [s]  (default 600.0)
            V_rc0       : float  initial RC overpotential [V] (default 0.0)
        """
        I = inputs.get("current_A", 20.0)
        soc0 = inputs.get("soc0", 0.9)
        T0 = inputs.get("T0_K", 593.15)
        dt = inputs.get("dt", 5.0)
        dur = inputs.get("duration_s", 600.0)
        V_rc0 = inputs.get("V_rc0", 0.0)

        return self._model.simulate(I, soc0, T0, dt, dur, V_rc0)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_A": {"unit": "A", "range": [-200.0, 200.0], "note": ">0 discharge, <0 charge"},
                "soc0": {"unit": "-", "range": [0.0, 1.0]},
                "T0_K": {"unit": "K", "range": [573.15, 623.15]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
                "V_rc0": {"unit": "V"},
            },
            "outputs": {
                "t": "s",
                "soc": "-",
                "voltage": "V",
                "current": "A",
                "power": "W",
                "temperature": "K",
                "v_rc": "V",
                "R0": "Ohm",
                "R1": "Ohm",
                "heat_gen": "W",
                "heater_power": "W",
                "functional": "bool",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"current_A": 30.0, "soc0": 0.9, "duration_s": 600.0, "dt": 10.0})
    print(
        f"Final SOC: {r['soc'][-1]:.4f}, "
        f"V: {r['voltage'][-1]:.4f} V, "
        f"T: {r['temperature'][-1]:.2f} K, "
        f"R0: {r['R0'][-1]*1000:.3f} mOhm, "
        f"heater: {r['heater_power'][-1]:.2f} W"
    )
