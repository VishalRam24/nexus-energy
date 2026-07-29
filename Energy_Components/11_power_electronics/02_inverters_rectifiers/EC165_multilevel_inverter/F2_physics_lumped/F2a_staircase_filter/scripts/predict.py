"""
EC165 -- Multilevel Inverter -- F2a Physics-Lumped (Staircase + Filter ODE)
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import MultilevelInverterF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC165 F2a physics-lumped multilevel inverter."""

    component_id = "EC165"
    component_name = "Multilevel Inverter (NPC / Cascaded H-Bridge)"
    fidelity = "F2a -- Physics-Lumped Staircase Synthesis + Averaged LC-Filter ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = MultilevelInverterF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a dynamic multilevel-inverter simulation.

        inputs:
            n_levels : int    (number of output levels, default from params)
            modulation_index : float  (m, default 1.0)
            n_periods : int   (fundamental periods to simulate, default 6)
            dt : float        (time step [s], default T/400)

        returns dict with time series (t, v_pole, i_L, v_out) and scalar metrics
        (thd_pole, thd_output, v_ac_rms, p_out, p_loss, efficiency, n_levels).
        """
        N = inputs.get("n_levels", self._model.n_levels)
        m = inputs.get("modulation_index", 1.0)
        n_periods = inputs.get("n_periods", 6)
        dt = inputs.get("dt", None)
        return self._model.simulate(m=m, n_levels=N, n_periods=n_periods, dt=dt)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "n_levels": {"unit": "-", "range": [2, 21]},
                "modulation_index": {"unit": "-", "range": [0.1, 1.15]},
                "n_periods": {"unit": "-"},
                "dt": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "v_pole": "V (switched staircase pole voltage)",
                "i_L": "A (filter inductor current)",
                "v_out": "V (filtered AC output, capacitor voltage)",
                "thd_pole": "- (fraction)",
                "thd_output": "- (fraction)",
                "v_ac_rms": "V (fundamental RMS)",
                "p_out": "W",
                "p_loss": "W",
                "efficiency": "-",
                "n_levels": "-",
            },
            "source": self._raw.get("unit", {}).get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"n_levels": 5, "modulation_index": 1.0, "n_periods": 4})
    print(f"5-level: THD_pole={r['thd_pole']*100:.2f}%  "
          f"V_ac_rms={r['v_ac_rms']:.1f} V  P_out={r['p_out']/1000:.2f} kW  "
          f"eff={r['efficiency']*100:.2f}%")
    r2 = m.predict({"n_levels": 9, "modulation_index": 1.0, "n_periods": 4})
    print(f"9-level: THD_pole={r2['thd_pole']*100:.2f}%  eff={r2['efficiency']*100:.2f}%")
