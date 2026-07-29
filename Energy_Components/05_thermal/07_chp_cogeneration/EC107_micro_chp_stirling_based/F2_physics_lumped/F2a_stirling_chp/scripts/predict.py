"""
EC107 -- Micro-CHP (Stirling-based) -- F2a Physics-Lumped
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import StirlingCHP_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the Stirling micro-CHP F2a physics model."""

    component_id = "EC107"
    component_name = "Micro-CHP (Stirling-based)"
    fidelity = "F2a -- Physics-Lumped Stirling micro-CHP with thermal warm-up ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = StirlingCHP_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a cold-start warm-up simulation of the Stirling micro-CHP.

        inputs:
            load_fraction : float in [0,1]  (burner firing fraction, default 1.0)
            T0_K          : float  (initial head temperature, default = T_amb)
            dt            : float  (output step [s], default 5.0)
            duration_s    : float  (total duration [s], default 1800.0)

        Returns dict of time-series arrays plus the converged steady-state
        split under key "steady_state".
        """
        load = inputs.get("load_fraction", 1.0)
        T0 = inputs.get("T0_K", self._model.T_amb)
        dt = inputs.get("dt", 5.0)
        dur = inputs.get("duration_s", 1800.0)

        result = self._model.simulate(load, T0, dt, dur)
        result["steady_state"] = self._model.steady_state(load)
        return result

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "load_fraction": {"unit": "-", "range": [0.0, 1.0]},
                "T0_K": {"unit": "K", "range": [273.15, 873.15]},
                "dt": {"unit": "s", "range": [0.1, 30.0]},
                "duration_s": {"unit": "s", "range": [1.0, 7200.0]},
            },
            "outputs": {
                "t": "s",
                "temperature": "K",
                "P_elec_W": "W",
                "Q_th_W": "W",
                "eta_elec": "-",
                "eta_th": "-",
                "eta_total": "-",
                "warmup_factor": "-",
                "steady_state": "dict of converged powers/efficiencies",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"load_fraction": 1.0, "dt": 30.0, "duration_s": 1800.0})
    ss = r["steady_state"]
    print(
        f"\nFinal head T: {r['temperature'][-1]:.1f} K | "
        f"P_elec: {r['P_elec_W'][-1]:.0f} W | Q_th: {r['Q_th_W'][-1]:.0f} W"
    )
    print(
        f"Steady state: eta_e={ss['eta_elec']:.3f}, eta_th={ss['eta_th']:.3f}, "
        f"eta_total={ss['eta_total']:.3f}, eta_Carnot={ss['eta_carnot']:.3f}, "
        f"P:H={ss['power_to_heat']:.3f}"
    )
