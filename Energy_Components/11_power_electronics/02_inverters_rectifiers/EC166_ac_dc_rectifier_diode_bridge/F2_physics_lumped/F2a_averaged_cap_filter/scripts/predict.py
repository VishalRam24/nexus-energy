"""
EC166 -- AC-DC Rectifier (Diode Bridge) -- F2a Averaged Cap-Filter
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import DiodeBridgeRectifierF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC166 F2a averaged diode-bridge model."""

    component_id = "EC166"
    component_name = "AC-DC Rectifier (Diode Bridge)"
    fidelity = "F2a -- Physics-Lumped Averaged Bridge with Output-Cap Voltage ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = DiodeBridgeRectifierF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the lumped capacitor-voltage simulation.

        inputs:
            v_ac_rms   : float  AC input (V_LL for 3-phase, V_phase for 1-phase)
            R_load     : float  DC load resistance [Ohm] (default from params)
            dt         : float  output step [s] (default 2e-5)
            duration_s : float  sim length [s] (default 0.1 = 5 line cycles @50Hz)
            v_C0       : float  initial cap voltage [V] (default ~0.95*Vdc_ideal)
        """
        v_ac = inputs.get("v_ac_rms", 400.0)
        R_load = inputs.get("R_load", None)
        dt = inputs.get("dt", 2e-5)
        dur = inputs.get("duration_s", 0.1)
        v_C0 = inputs.get("v_C0", None)
        return self._model.simulate(v_ac, R_load, dt, dur, v_C0)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "v_ac_rms": {"unit": "V", "range": [100, 690]},
                "R_load": {"unit": "Ohm", "range": [1, 1000]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
                "v_C0": {"unit": "V"},
            },
            "outputs": {
                "t": "s",
                "v_dc": "V (capacitor / DC-bus voltage)",
                "i_load": "A",
                "i_diode": "A (pulsed charging current)",
                "v_dc_mean": "V",
                "v_ripple_pp": "V",
                "efficiency": "-",
                "power_factor": "-",
                "v_dc_ideal": "V",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"v_ac_rms": 400.0, "R_load": 8.0, "duration_s": 0.1})
    print(
        f"V_dc_mean = {r['v_dc_mean']:.1f} V (ideal {r['v_dc_ideal']:.1f} V), "
        f"ripple_pp = {r['v_ripple_pp']:.2f} V, eff = {r['efficiency']:.3f}, "
        f"PF = {r['power_factor']:.3f}"
    )
