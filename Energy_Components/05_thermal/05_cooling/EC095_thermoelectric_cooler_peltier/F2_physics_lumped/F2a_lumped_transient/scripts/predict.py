"""
EC095 — Thermoelectric Cooler (Peltier) — F2a Physics-Lumped Transient
Standardised predict() / get_info() interface (mirrors EC001 F2a).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import PeltierTEC_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


def _to_K(x):
    return x + 273.15


class ComponentModel:
    """Standardised wrapper for the EC095 Peltier F2a transient lumped model."""

    component_id = "EC095"
    component_name = "Thermoelectric Cooler (Peltier)"
    fidelity = "F2a — Physics-Lumped Transient (Seebeck + Joule + Fourier, 2-node ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = PeltierTEC_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the transient lumped simulation.

        inputs (all optional, degrees Celsius for temperatures):
            current_A     : float (per-module current, series stack)   default 4.0
            T_cold0_C     : float initial cold-plate temp              default = T_ambient_C
            T_hot0_C      : float initial hot-plate temp               default = T_ambient_C
            T_load_C      : float cold-side load reservoir temp        default 25.0
            T_ambient_C   : float ambient temp                         default 25.0
            Q_load_W      : float external heat load on cold side      default 0.0
            dt            : float output interval [s]                  default 1.0
            duration_s    : float total time [s]                       default 600.0
        """
        I = inputs.get("current_A", 4.0)
        T_amb_C = inputs.get("T_ambient_C", 25.0)
        T_load_C = inputs.get("T_load_C", 25.0)
        Tc0_C = inputs.get("T_cold0_C", T_amb_C)
        Th0_C = inputs.get("T_hot0_C", T_amb_C)
        Q_load = inputs.get("Q_load_W", 0.0)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 600.0)

        return self._model.simulate(
            I, _to_K(Tc0_C), _to_K(Th0_C), _to_K(T_load_C), _to_K(T_amb_C),
            Q_load_W=Q_load, dt=dt, duration_s=dur,
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_A": {"unit": "A", "range": [0, 6.0]},
                "T_cold0_C": {"unit": "degC"},
                "T_hot0_C": {"unit": "degC"},
                "T_load_C": {"unit": "degC", "range": [-20, 40]},
                "T_ambient_C": {"unit": "degC", "range": [0, 50]},
                "Q_load_W": {"unit": "W", "range": [0, 300]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_cold": "K", "T_hot": "K",
                "T_cold_C": "degC", "T_hot_C": "degC",
                "current": "A",
                "Q_cold": "W (heat pumped from cold side)",
                "Q_hot": "W (heat rejected hot side)",
                "W_elec": "W (electrical input)",
                "cop": "- (cooling COP)",
                "cop_carnot": "- (Carnot bound)",
                "dT": "K (T_hot - T_cold)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"current_A": 4.0, "T_ambient_C": 25.0, "T_load_C": 10.0,
                   "Q_load_W": 20.0, "duration_s": 600.0, "dt": 5.0})
    print(f"Final T_cold: {r['T_cold_C'][-1]:.2f} C, T_hot: {r['T_hot_C'][-1]:.2f} C, "
          f"COP: {r['cop'][-1]:.3f} (Carnot {r['cop_carnot'][-1]:.2f})")
