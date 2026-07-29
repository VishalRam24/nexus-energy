"""
EC076 -- Regenerative Heat Exchanger -- F2a Physics-Lumped Regenerator Matrix
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import RegeneratorF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC076 F2a periodic-flow regenerator model."""

    component_id = "EC076"
    component_name = "Regenerative Heat Exchanger"
    fidelity = "F2a -- Physics-Lumped Periodic-Flow / Rotary Regenerator Matrix (N-node ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = RegeneratorF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the periodic-flow regenerator simulation to periodic steady state.

        inputs:
            T_h_in_K : float   hot-stream inlet temperature (K), default 573.15
            T_c_in_K : float   cold-stream inlet temperature (K), default 293.15
            C_h : float        hot-stream heat capacity rate (W/K), optional
            C_c : float        cold-stream heat capacity rate (W/K), optional
            rpm : float        wheel rotational speed (rev/min), optional
            n_cycles : int     max hot/cold cycles to periodic steady state (default 60)

        returns dict: outlet temperatures, effectiveness (ODE + correlation),
                      heat duty, NTU_o, matrix capacity ratio Cr*.
        """
        T_h_in = inputs.get("T_h_in_K", 573.15)
        T_c_in = inputs.get("T_c_in_K", 293.15)
        C_h = inputs.get("C_h", None)
        C_c = inputs.get("C_c", None)
        rpm = inputs.get("rpm", None)
        n_cycles = int(inputs.get("n_cycles", 60))

        return self._model.simulate(T_h_in, T_c_in, n_cycles=n_cycles,
                                    C_h=C_h, C_c=C_c, rpm=rpm)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_h_in_K": {"unit": "K", "range": [323.15, 873.15]},
                "T_c_in_K": {"unit": "K", "range": [253.15, 373.15]},
                "C_h": {"unit": "W/K", "range": [200, 20000]},
                "C_c": {"unit": "W/K", "range": [200, 20000]},
                "rpm": {"unit": "rev/min", "range": [0.5, 60]},
                "n_cycles": {"unit": "-"},
            },
            "outputs": {
                "T_h_out": "K",
                "T_c_out": "K",
                "effectiveness_ode": "-",
                "effectiveness_correlation": "-",
                "Q_kW": "kW",
                "NTU_o": "-",
                "Cr_star": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"T_h_in_K": 573.15, "T_c_in_K": 293.15})
    print(f"eps(ODE)={r['effectiveness_ode']:.4f}  "
          f"eps(corr)={r['effectiveness_correlation']:.4f}  "
          f"T_c_out={r['T_c_out']:.2f} K  Q={r['Q_kW']:.2f} kW  "
          f"Cr*={r['Cr_star']:.2f}  NTU_o={r['NTU_o']:.2f}  "
          f"(cycles={r['n_cycles_run']})")
