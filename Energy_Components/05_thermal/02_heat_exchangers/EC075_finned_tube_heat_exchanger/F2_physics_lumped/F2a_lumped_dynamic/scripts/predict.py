"""
EC075 -- Finned-Tube Heat Exchanger -- F2a Physics-Lumped (Transient)
Standardised predict() / get_info() interface (mirrors EC001 F2a).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import FinnedTubeHXF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the finned-tube HX F2a lumped transient model."""

    component_id = "EC075"
    component_name = "Finned-Tube Heat Exchanger"
    fidelity = "F2a -- Physics-Lumped Transient (multi-CV energy balance)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = FinnedTubeHXF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a transient HX simulation under constant inlet/flow conditions and
        return the time-series approach to steady state plus steady outlets.

        inputs:
            T_h_in      : float   hot (water) inlet temperature [degC]   (default 80)
            T_c_in      : float   cold (air) inlet temperature  [degC]   (default 20)
            m_dot_hot   : float   hot mass flow  [kg/s]                  (default 1.0)
            m_dot_cold  : float   cold mass flow [kg/s]                  (default 2.0)
            duration_s  : float   simulation horizon [s]                 (default 600)
            n_out       : int     number of output samples               (default 120)
            T0          : float   initial uniform temperature [degC]     (default T_c_in)
        """
        T_h_in = float(inputs.get("T_h_in", 80.0))
        T_c_in = float(inputs.get("T_c_in", 20.0))
        m_dot_hot = float(inputs.get("m_dot_hot", 1.0))
        m_dot_cold = float(inputs.get("m_dot_cold", 2.0))
        duration_s = float(inputs.get("duration_s", 600.0))
        n_out = int(inputs.get("n_out", 120))
        T0 = inputs.get("T0", None)

        r = self._model.simulate(T_h_in, T_c_in, m_dot_hot, m_dot_cold,
                                 duration_s=duration_s, n_out=n_out,
                                 T0=None if T0 is None else float(T0))

        # Scalar steady-state summary (final sample) + e-NTU reference.
        ref = self._model.steady_state_entu(T_h_in, T_c_in, m_dot_hot, m_dot_cold)
        r["steady_state"] = {
            "Q_kw": float(r["Q_kw"][-1]),
            "T_h_out": float(r["T_h_out"][-1]),
            "T_c_out": float(r["T_c_out"][-1]),
            "effectiveness": float(r["effectiveness"][-1]),
        }
        r["entu_reference"] = ref
        return r

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "ec_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_h_in": {"unit": "degC", "range": [20.0, 120.0]},
                "T_c_in": {"unit": "degC", "range": [-20.0, 50.0]},
                "m_dot_hot": {"unit": "kg/s", "range": [0.05, 5.0]},
                "m_dot_cold": {"unit": "kg/s", "range": [0.1, 10.0]},
                "duration_s": {"unit": "s", "range": [1.0, 7200.0]},
            },
            "outputs": {
                "t": "s",
                "T_h_out": "degC",
                "T_c_out": "degC",
                "T_wall_mean": "degC",
                "Q_kw": "kW",
                "effectiveness": "-",
                "steady_state": "dict of scalar steady outlets",
                "entu_reference": "dict, analytic e-NTU steady state",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                   "m_dot_hot": 1.0, "m_dot_cold": 2.0, "duration_s": 600.0})
    ss = r["steady_state"]
    ref = r["entu_reference"]
    print(f"Steady: Q={ss['Q_kw']:.2f} kW, eps={ss['effectiveness']:.4f}, "
          f"T_h_out={ss['T_h_out']:.2f} C, T_c_out={ss['T_c_out']:.2f} C")
    print(f"e-NTU : Q={ref['Q_kw']:.2f} kW, eps={ref['effectiveness']:.4f}")
