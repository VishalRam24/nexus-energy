"""
EC099 -- Stirling Engine -- F2a Physics-Lumped Ideal Cycle
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import StirlingEngineF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the Stirling F2a physics-lumped model."""

    component_id = "EC099"
    component_name = "Stirling Engine"
    fidelity = "F2a -- Physics-Lumped Schmidt Ideal Cycle + Warm-up ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["engine"].update(params)
        self._model = StirlingEngineF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a warm-up transient and report cycle performance.

        inputs:
            T_h0 : float       initial hot-end temperature [K] (default ambient)
            T_c  : float       cold-side temperature [K]
            n_rpm: float       engine speed [rpm]
            p_mean: float      mean charge pressure [Pa]
            Q_burner: float    burner duty [W]
            dt : float         output time step [s] (default 1.0)
            duration_s : float horizon [s] (default 600.0)
        """
        T_h0 = inputs.get("T_h0", None)
        T_c = inputs.get("T_c", None)
        n_rpm = inputs.get("n_rpm", None)
        p_mean = inputs.get("p_mean", None)
        Q_burner = inputs.get("Q_burner", None)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 600.0)

        return self._model.simulate(T_h0, T_c, n_rpm, p_mean, Q_burner, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_h0": {"unit": "K", "range": [293.15, 1100.0]},
                "T_c": {"unit": "K", "range": [278.0, 360.0]},
                "n_rpm": {"unit": "rpm", "range": [100.0, 4000.0]},
                "p_mean": {"unit": "Pa", "range": [1.0e5, 1.5e7]},
                "Q_burner": {"unit": "W"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_h": "K (hot-end temperature)",
                "indicated_power": "W",
                "brake_power": "W",
                "efficiency": "- (indicated thermal)",
                "carnot_eff": "-",
                "beale_power": "W (cross-check)",
                "heat_input": "W",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"duration_s": 600.0, "dt": 10.0})
    print(f"Final hot-end T: {r['T_h'][-1]:.1f} K")
    print(f"Indicated power: {r['indicated_power'][-1]:.1f} W")
    print(f"Brake power:     {r['brake_power'][-1]:.1f} W")
    print(f"Efficiency:      {r['efficiency'][-1]*100:.2f} %  "
          f"(Carnot {r['carnot_eff'][-1]*100:.2f} %)")
    print(f"Beale cross-check: {r['beale_power'][-1]:.1f} W")
