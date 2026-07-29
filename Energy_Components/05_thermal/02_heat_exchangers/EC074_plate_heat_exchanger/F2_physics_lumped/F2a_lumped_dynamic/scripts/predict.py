"""
EC074 -- Plate Heat Exchanger -- F2a Lumped Dynamic
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import PlateHeatExchanger_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for Plate Heat Exchanger F2a lumped dynamic model."""

    component_id = "EC074"
    component_name = "Plate Heat Exchanger"
    fidelity = "F2a — Lumped Dynamic Two-Fluid Capacitance Model"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = PlateHeatExchanger_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation.

        inputs:
            m_dot_hot : float [kg/s] (default 1.2)
            m_dot_cold : float [kg/s] (default 1.0)
            T_hot_in : float [K] (default 353.15)
            T_cold_in : float [K] (default 293.15)
            T_hot_init : float [K] (default 293.15)
            T_cold_init : float [K] (default 293.15)
            dt : float [s] (default 1.0)
            duration_s : float [s] (default 300.0)
        """
        m_dot_h = inputs.get("m_dot_hot", 1.2)
        m_dot_c = inputs.get("m_dot_cold", 1.0)
        T_h_in = inputs.get("T_hot_in", 353.15)
        T_c_in = inputs.get("T_cold_in", 293.15)
        T_h_init = inputs.get("T_hot_init", 293.15)
        T_c_init = inputs.get("T_cold_init", 293.15)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 300.0)

        return self._model.simulate(m_dot_h, m_dot_c, T_h_in, T_c_in,
                                    T_h_init, T_c_init, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "m_dot_hot": {"unit": "kg/s", "range": [0, 3.0]},
                "m_dot_cold": {"unit": "kg/s", "range": [0, 3.0]},
                "T_hot_in": {"unit": "K", "range": [293.15, 423.15]},
                "T_cold_in": {"unit": "K", "range": [273.15, 373.15]},
                "T_hot_init": {"unit": "K"},
                "T_cold_init": {"unit": "K"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_hot_out": "K",
                "T_cold_out": "K",
                "Q_transfer": "W",
                "effectiveness": "-",
                "UA": "W/K",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"duration_s": 60.0, "dt": 1.0})
    print(f"Final T_hot_out: {r['T_hot_out'][-1]:.2f} K, T_cold_out: {r['T_cold_out'][-1]:.2f} K")
