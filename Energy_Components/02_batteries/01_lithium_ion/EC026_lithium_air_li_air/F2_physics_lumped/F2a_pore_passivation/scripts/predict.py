"""
EC026 -- Lithium-Air Battery (Li-O2) -- F2a Physics-Lumped (Pore Passivation)
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import LiAirF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the Li-Air F2a pore-passivation model."""

    component_id = "EC026"
    component_name = "Lithium-Air Battery (Li-O2 / Li-Air)"
    fidelity = "F2a -- Physics-Lumped Pore-Passivation Electrochemical/Thevenin Hybrid with Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = LiAirF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation.

        inputs:
            current_A   : float or callable(t)  (+ = discharge, - = charge), default 0.5
            soc_0       : float  initial state of charge, default 1.0
            theta_0     : float  initial Li2O2 pore-fill fraction, default 0.0
            T0_K        : float  initial temperature [K], default ambient
            dt          : float  output step [s], default 10.0
            duration_s  : float  total duration [s], default 3600.0
        """
        I = inputs.get("current_A", 0.5)
        soc0 = inputs.get("soc_0", 1.0)
        theta0 = inputs.get("theta_0", 0.0)
        T0 = inputs.get("T0_K", None)
        dt = inputs.get("dt", 10.0)
        dur = inputs.get("duration_s", 3600.0)
        return self._model.simulate(I, soc0, theta0, T0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_A": {"unit": "A", "range": [-2.0, 2.0], "note": "+discharge / -charge"},
                "soc_0": {"unit": "-", "range": [0.0, 1.0]},
                "theta_0": {"unit": "-", "range": [0.0, 1.0], "note": "Li2O2 pore-fill fraction"},
                "T0_K": {"unit": "K", "range": [258.15, 333.15]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "voltage": "V",
                "current": "A",
                "soc": "-",
                "theta": "- (pore fill)",
                "equilibrium_voltage": "V",
                "power": "W",
                "efficiency": "- (voltaic)",
                "temperature": "K",
                "overpotentials": "dict of V arrays",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    # discharge a fresh cell at 1 A until pore saturation cuts it off
    r = m.predict({"current_A": 1.0, "soc_0": 1.0, "duration_s": 3600.0, "dt": 30.0})
    print(f"Discharge: V0={r['voltage'][0]:.3f} V  V_end={r['voltage'][-1]:.3f} V  "
          f"theta_end={r['theta'][-1]:.3f}  T_end={r['temperature'][-1]:.2f} K")
    Vd, Vc, gap = m._model.round_trip_voltage_gap(1.0, soc=0.5, theta=0.1)
    print(f"Round-trip gap @1A: V_dis={Vd:.3f}  V_chg={Vc:.3f}  gap={gap:.3f} V")
