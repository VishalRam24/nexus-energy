"""
EC161 -- Dual Active Bridge (DAB) DC-DC Converter -- F2a Phase-Shift Averaged
Standardised predict() / get_info() interface (mirrors EC001 F2a).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import DAB_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the DAB F2a phase-shift averaged model."""

    component_id = "EC161"
    component_name = "Dual Active Bridge (DAB) DC-DC Converter"
    fidelity = "F2a -- Phase-Shift Averaged Physics-Lumped Model"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = DAB_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run averaged dynamic simulation of the DAB output bus.

        inputs:
            phi        : float, phase shift [rad] (sign = power direction).
                         If omitted, derived from p_target via SPS inverse.
            p_target   : float, requested power [W] (alternative to phi).
            v_in       : float, primary bus voltage [V] (default nominal).
            v_out_0    : float, initial output-bus voltage [V] (default nominal).
            r_load     : float, output resistive load [Ohm] (default nominal).
            dt         : float, output sample spacing [s] (default 2e-5).
            duration_s : float, total simulated time [s] (default 2e-3).
        """
        v_in = inputs.get("v_in", self._model.V1_nom)
        v_out_0 = inputs.get("v_out_0", self._model.V2_nom)
        r_load = inputs.get("r_load", self._model.R_load)
        dt = inputs.get("dt", 2e-5)
        dur = inputs.get("duration_s", 2e-3)

        if "phi" in inputs:
            phi = inputs["phi"]
        elif "p_target" in inputs:
            phi = float(self._model.phase_for_power(v_in, v_out_0, inputs["p_target"]))
        else:
            phi = float(self._model.phase_for_power(v_in, v_out_0,
                                                    self._model.power_max(v_in, v_out_0) * 0.5))

        return self._model.simulate(phi, v1=v_in, v2_0=v_out_0,
                                    r_load=r_load, dt=dt, duration_s=dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "phi": {"unit": "rad", "range": [-1.5707963, 1.5707963]},
                "p_target": {"unit": "W", "range": [-15000, 15000]},
                "v_in": {"unit": "V", "range": [200, 800]},
                "v_out_0": {"unit": "V", "range": [100, 400]},
                "r_load": {"unit": "Ohm"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "v_out": "V",
                "phi": "rad",
                "power_transfer": "W",
                "power_loss": "W",
                "efficiency": "-",
                "i_rms": "A",
                "full_zvs": "bool",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"p_target": 5000.0, "duration_s": 2e-3, "dt": 2e-5})
    print(f"phi={r['phi'][0]:.4f} rad  V_out: {r['v_out'][0]:.2f} -> "
          f"{r['v_out'][-1]:.2f} V  P={r['power_transfer'][-1]:.1f} W  "
          f"eta={r['efficiency'][-1]:.4f}  ZVS={bool(r['full_zvs'][-1])}")
