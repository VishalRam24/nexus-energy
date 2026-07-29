"""
EC189 -- Natural Gas Pipeline -- F2a Physics-Lumped Line-Pack Dynamics
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import NGPipelineF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the NG pipeline F2a line-pack dynamics model."""

    component_id = "EC189"
    component_name = "Natural Gas Pipeline"
    fidelity = "F2a -- Physics-Lumped Isothermal Line-Pack Dynamics (ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = NGPipelineF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a transient line-pack simulation.

        inputs:
            P_avg0_bar  : float  initial mean pipe pressure (default 60.0)
            P_out_bar   : float  fixed delivery pressure      (default 50.0)
            m_in_kg_s   : float or callable(t)  supply inflow (default 100.0)
            P_in_bar    : float or None  if set, drive inflow from upstream P
            dt          : float  output step [s]              (default 60.0)
            duration_s  : float  total duration [s]           (default 3600.0)
        """
        P_avg0 = inputs.get("P_avg0_bar", 60.0)
        P_out = inputs.get("P_out_bar", 50.0)
        m_in = inputs.get("m_in_kg_s", 100.0)
        P_in = inputs.get("P_in_bar", None)
        dt = inputs.get("dt", 60.0)
        dur = inputs.get("duration_s", 3600.0)

        return self._model.simulate(P_avg0, P_out, m_in, dt, dur, P_in_bar=P_in)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "P_avg0_bar": {"unit": "bar", "range": [5, 150]},
                "P_out_bar": {"unit": "bar", "range": [1, 140]},
                "m_in_kg_s": {"unit": "kg/s", "note": "float or callable(t)"},
                "P_in_bar": {"unit": "bar", "range": [5, 150], "note": "optional driven mode"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "P_avg": "bar",
                "linepack_mass": "kg",
                "m_in": "kg/s",
                "m_out": "kg/s",
                "Q_out": "std m3/day",
                "friction_factor": "-",
                "reynolds": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    # Supply exceeds outflow -> pipe packs up (P_avg rises)
    r = m.predict({"P_avg0_bar": 55.0, "P_out_bar": 50.0,
                   "m_in_kg_s": 150.0, "dt": 60.0, "duration_s": 3600.0})
    print(f"P_avg: {r['P_avg'][0]:.2f} -> {r['P_avg'][-1]:.2f} bar | "
          f"line-pack: {r['linepack_mass'][0]/1e3:.1f} -> "
          f"{r['linepack_mass'][-1]/1e3:.1f} tonnes | "
          f"m_out final: {r['m_out'][-1]:.1f} kg/s")
