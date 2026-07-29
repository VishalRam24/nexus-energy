"""
EC034 -- Aluminum-Ion Battery -- F2a Thevenin ECM (1-RC/2-RC)
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import AluminumIonECM_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the Al-ion F2a Thevenin ECM model."""

    component_id = "EC034"
    component_name = "Aluminum-Ion Battery"
    fidelity = "F2a -- Thevenin ECM (1-RC/2-RC) + Coulomb SOC + thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = AluminumIonECM_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation.

        inputs:
            current_A   : float (or callable(t)) load current; >0 discharge, <0 charge
            soc0        : float initial state of charge (default 0.9)
            T_cell_K    : float initial temperature (default ambient)
            dt          : float output step [s] (default 1.0)
            duration_s  : float total time [s] (default 600.0)
            n_rc        : int   1 or 2 RC pairs (optional override)
        """
        if "n_rc" in inputs:
            self._model.n_rc = int(inputs["n_rc"])

        I = inputs.get("current_A", 1.0)
        soc0 = inputs.get("soc0", 0.9)
        T0 = inputs.get("T_cell_K", None)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 600.0)

        return self._model.simulate(I, soc0=soc0, T0=T0, dt=dt, duration_s=dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_A": {"unit": "A", "range": [-20.0, 20.0],
                              "note": ">0 discharge, <0 charge"},
                "soc0": {"unit": "-", "range": [0.0, 1.0]},
                "T_cell_K": {"unit": "K", "range": [263.15, 333.15]},
                "dt": {"unit": "s", "range": [0.01, 10.0]},
                "duration_s": {"unit": "s", "range": [1.0, 36000.0]},
                "n_rc": {"unit": "-", "range": [1, 2]},
            },
            "outputs": {
                "t": "s",
                "soc": "-",
                "voltage": "V",
                "ocv": "V",
                "current": "A",
                "power": "W",
                "temperature": "K",
                "efficiency": "-",
                "heat_gen": "W",
                "v_rc": "dict of V arrays",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"current_A": 2.0, "soc0": 0.9, "dt": 5.0, "duration_s": 300.0})
    print(f"SOC: {r['soc'][0]:.3f} -> {r['soc'][-1]:.3f} | "
          f"V0={r['voltage'][0]:.3f} V, Vend={r['voltage'][-1]:.3f} V | "
          f"T: {r['temperature'][0]:.2f} -> {r['temperature'][-1]:.2f} K")
