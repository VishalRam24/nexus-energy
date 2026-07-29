"""
EC039 -- Organic Flow Battery (OFB) -- F2a Physics-Lumped Stack Model
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import OrganicFlowF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC039 organic flow battery F2a stack model."""

    component_id = "EC039"
    component_name = "Organic Flow Battery (OFB)"
    fidelity = "F2a -- Physics-Lumped Stack (electrochemical + SOC + fade + thermal ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = OrganicFlowF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run coupled SOC / capacity-fade / thermal simulation.

        inputs:
            current_A   : float or list (stack current; +discharge, -charge), default 20.0
            soc0        : float (initial SOC), default 0.9
            T0_K        : float (initial temperature), default 298.15
            cap0        : float (initial capacity fraction), default 1.0
            dt          : float (output step, s), default 5.0
            duration_s  : float (total time, s), default 600.0
        """
        I = inputs.get("current_A", 20.0)
        soc0 = inputs.get("soc0", 0.9)
        T0 = inputs.get("T0_K", 298.15)
        cap0 = inputs.get("cap0", 1.0)
        dt = inputs.get("dt", 5.0)
        dur = inputs.get("duration_s", 600.0)
        return self._model.simulate(I, soc0, T0, dt, dur, cap0=cap0)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_A": {"unit": "A", "range": [-50.0, 50.0],
                              "note": "+discharge / -charge"},
                "soc0": {"unit": "-", "range": [0.05, 0.95]},
                "T0_K": {"unit": "K", "range": [283.15, 313.15]},
                "cap0": {"unit": "-", "range": [0.0, 1.0]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "soc": "-",
                "capacity": "fraction",
                "voltage": "V (stack)",
                "cell_voltage": "V",
                "power": "W",
                "temperature": "K",
                "efficiency": "- (round-trip energy)",
                "overpotentials": "dict of V arrays",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"current_A": 20.0, "soc0": 0.9, "duration_s": 600.0, "dt": 30.0})
    print(f"Final SOC: {r['soc'][-1]:.4f}, "
          f"Final stack V: {r['voltage'][-1]:.3f} V, "
          f"Final T: {r['temperature'][-1]:.3f} K, "
          f"Energy eff: {r['efficiency'][-1]:.4f}")
