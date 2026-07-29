"""
EC025 -- Lithium-Sulfur Battery (Li-S) -- F2a Two-Plateau + Shuttle + Thermal
Standardised predict() / get_info() interface (mirrors EC001 F2a template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import LiS_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for Li-S F2a two-plateau + polysulfide-shuttle model."""

    component_id = "EC025"
    component_name = "Lithium-Sulfur Battery (Li-S)"
    fidelity = "F2a -- Two-Plateau 0D Electrochemical/Thevenin + Polysulfide Shuttle + Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = LiS_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a dynamic Li-S simulation.

        inputs:
            current_A : float (or callable)  applied current [A], I>0 discharge
            soc0 : float        initial SOC (default 1.0)
            T0 : float          initial temperature [K] (default 298.15)
            dt : float          output step [s] (default 5.0)
            duration_s : float  total duration [s] (default 3600.0)
            V_rc0 : float       initial RC overpotential [V] (default 0.0)
        """
        I = inputs.get("current_A", 1.0)
        soc0 = inputs.get("soc0", 1.0)
        T0 = inputs.get("T0", 298.15)
        dt = inputs.get("dt", 5.0)
        dur = inputs.get("duration_s", 3600.0)
        V_rc0 = inputs.get("V_rc0", 0.0)
        return self._model.simulate(I, soc0, T0, dt, dur, V_rc0=V_rc0)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_A": {"unit": "A", "range": [-6.0, 6.0], "note": "I>0 discharge"},
                "soc0": {"unit": "-", "range": [0.0, 1.0]},
                "T0": {"unit": "K", "range": [258.15, 333.15]},
                "dt": {"unit": "s", "range": [0.1, 60.0]},
                "duration_s": {"unit": "s", "range": [1.0, 36000.0]},
                "V_rc0": {"unit": "V"},
            },
            "outputs": {
                "t": "s",
                "soc": "-",
                "voltage": "V",
                "ocv": "V",
                "temperature": "K",
                "v_rc": "V",
                "shuttle_current": "A",
                "coulombic_efficiency": "-",
                "power": "W",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"current_A": 1.0, "soc0": 1.0, "dt": 30.0, "duration_s": 3600.0})
    print(
        f"After 1h @ 1A: SOC={r['soc'][-1]:.3f}, V={r['voltage'][-1]:.3f} V, "
        f"T={r['temperature'][-1]:.2f} K, eta_C={r['coulombic_efficiency'][-1]:.3f}, "
        f"I_shuttle={r['shuttle_current'][-1]:.4f} A"
    )
