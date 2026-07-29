"""
EC077 -- Microchannel Heat Exchanger -- F2a Lumped Transient
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import MicrochannelHX_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the MCHX F2a lumped transient model."""

    component_id = "EC077"
    component_name = "Microchannel Heat Exchanger (MCHX)"
    fidelity = "F2a -- Lumped Transient (N-CV two-stream + wall energy ODEs)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = MicrochannelHX_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a transient simulation to (near) steady state.

        inputs:
            T_h_in : float [degC]    hot inlet temperature
            T_c_in : float [degC]    cold inlet temperature
            mdot_h : float [kg/s]    hot mass flow
            mdot_c : float [kg/s]    cold mass flow
            hot_stream / cold_stream : "hot" | "cold" | "air" (property set)
            dt : float [s]           output step
            duration_s : float [s]   simulation horizon

        Returns time-series dict + steady profiles, duty, effectiveness, dP.
        """
        T_h_in = inputs.get("T_h_in", self._model.T_h_in0)
        T_c_in = inputs.get("T_c_in", self._model.T_c_in0)
        mdot_h = inputs.get("mdot_h", self._model.mdot_h0)
        mdot_c = inputs.get("mdot_c", self._model.mdot_c0)
        hot_stream = inputs.get("hot_stream", "hot")
        cold_stream = inputs.get("cold_stream", "cold")
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 120.0)

        return self._model.simulate(
            T_h_in=T_h_in, T_c_in=T_c_in, mdot_h=mdot_h, mdot_c=mdot_c,
            dt=dt, duration_s=dur,
            hot_stream=hot_stream, cold_stream=cold_stream,
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_h_in": {"unit": "degC", "range": [30, 150]},
                "T_c_in": {"unit": "degC", "range": [-20, 40]},
                "mdot_h": {"unit": "kg/s", "range": [0.005, 0.5]},
                "mdot_c": {"unit": "kg/s", "range": [0.005, 0.5]},
                "hot_stream": {"unit": "-", "options": ["hot", "cold", "air"]},
                "cold_stream": {"unit": "-", "options": ["hot", "cold", "air"]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_h_out": "degC",
                "T_c_out": "degC",
                "Q_kW": "kW",
                "effectiveness": "-",
                "T_h_profile": "degC (per node, steady)",
                "T_c_profile": "degC (per node, steady)",
                "T_wall_profile": "degC (per node, steady)",
                "dP_h_Pa": "Pa",
                "dP_c_Pa": "Pa",
                "Re_h": "-", "Re_c": "-",
                "h_h": "W/(m2.K)", "h_c": "W/(m2.K)",
                "UA": "W/K",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"T_h_in": 80.0, "T_c_in": 20.0, "duration_s": 120.0, "dt": 5.0})
    print(f"Steady: T_h_out={r['T_h_out'][-1]:.2f} C, "
          f"T_c_out={r['T_c_out'][-1]:.2f} C, Q={r['Q_kW'][-1]:.3f} kW, "
          f"eps={r['effectiveness'][-1]:.4f}, UA={r['UA']:.1f} W/K, "
          f"h_h={r['h_h']:.0f} W/m2K, dP_h={r['dP_h_Pa']:.0f} Pa, Re_h={r['Re_h']:.1f}")
