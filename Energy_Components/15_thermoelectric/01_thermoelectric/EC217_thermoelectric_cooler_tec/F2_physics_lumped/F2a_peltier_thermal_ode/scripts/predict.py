"""
EC217 -- Thermoelectric Cooler (TEC) -- F2a Peltier + Lumped Thermal ODE
Standardised predict() / get_info() interface (mirrors EC001 template).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import TEC_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the TEC F2a Peltier + thermal-ODE model."""

    component_id = "EC217"
    component_name = "Thermoelectric Cooler (TEC)"
    fidelity = "F2a -- Peltier Cooler with Lumped Cold/Hot-Plate Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = TEC_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the dynamic pull-down simulation.

        inputs:
            current_A    : float  drive current [A] (default 3.0)
            T_cold0_K    : float  initial cold-plate temp [K] (default ambient)
            T_hot0_K     : float  initial hot-plate temp [K]  (default ambient)
            Q_load_W     : float  active heat load on cold plate [W] (default 0)
            duration_s   : float  horizon [s] (default 120)
            n_eval       : int    number of output points (default 200)
        """
        I = inputs.get("current_A", 3.0)
        T_c0 = inputs.get("T_cold0_K", None)
        T_h0 = inputs.get("T_hot0_K", None)
        Q_load = inputs.get("Q_load_W", None)
        dur = inputs.get("duration_s", 120.0)
        n_eval = inputs.get("n_eval", 200)

        return self._model.simulate(
            I, T_cold0=T_c0, T_hot0=T_h0, Q_load=Q_load,
            duration_s=dur, n_eval=n_eval,
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_A": {"unit": "A", "range": [0.0, 12.0]},
                "T_cold0_K": {"unit": "K", "range": [200.0, 320.0]},
                "T_hot0_K": {"unit": "K", "range": [280.0, 400.0]},
                "Q_load_W": {"unit": "W", "range": [0.0, 50.0]},
                "duration_s": {"unit": "s", "range": [1.0, 3600.0]},
                "n_eval": {"unit": "-"},
            },
            "outputs": {
                "t": "s",
                "T_cold": "K",
                "T_hot": "K",
                "Q_cold_W": "W (cooling power)",
                "Q_hot_W": "W (heat rejected)",
                "W_input_W": "W (electrical input)",
                "COP": "- (coefficient of performance)",
                "V_module_V": "V",
                "steady_state": "dict (Q_cold, COP, T_cold_ss, dT_ss, carnot_COP, ZT_avg)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"current_A": 3.0, "Q_load_W": 2.0, "duration_s": 120.0})
    ss = r["steady_state"]
    print(f"Steady-state: T_cold={ss['T_cold_ss_K']:.2f} K, "
          f"T_hot={ss['T_hot_ss_K']:.2f} K, dT={ss['dT_ss_K']:.2f} K, "
          f"Q_cold={ss['Q_cold_W']:.3f} W, COP={ss['COP']:.3f} "
          f"(Carnot {ss['carnot_COP']:.3f}), ZT={ss['ZT_avg']:.3f}")
