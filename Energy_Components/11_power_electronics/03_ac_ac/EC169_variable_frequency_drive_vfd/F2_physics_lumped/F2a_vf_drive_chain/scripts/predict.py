"""
EC169 -- Variable Frequency Drive (VFD) -- F2a Physics-Lumped V/f Drive Chain
Standardised predict() / get_info() interface (mirrors EC001 F2a template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import VFDF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the VFD F2a physics-lumped V/f drive-chain model."""

    component_id = "EC169"
    component_name = "Variable Frequency Drive (VFD)"
    fidelity = "F2a -- Physics-Lumped V/f Drive Chain (DC-link + motor-speed ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = VFDF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the coupled DC-link + motor-speed dynamic simulation.

        inputs:
            f_set      : float or callable(t)->Hz   inverter output frequency (default 40)
            T_load     : float or callable(t)->N.m  load torque (default 50)
            V_dc0      : float  initial DC-link voltage [V] (default nominal)
            omega_m0   : float  initial rotor speed [rad/s] (default 0)
            dt         : float  output step [s] (default 0.005)
            duration_s : float  horizon [s] (default 3.0)
        """
        f_set = inputs.get("f_set", 40.0)
        T_load = inputs.get("T_load", 50.0)
        V_dc0 = inputs.get("V_dc0", None)
        omega_m0 = inputs.get("omega_m0", 0.0)
        dt = inputs.get("dt", 0.005)
        dur = inputs.get("duration_s", 3.0)

        return self._model.simulate(f_set, T_load, V_dc0, omega_m0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "f_set": {"unit": "Hz", "range": [0, 120]},
                "T_load": {"unit": "N.m", "range": [0, 200]},
                "V_dc0": {"unit": "V", "range": [0, 800]},
                "omega_m0": {"unit": "rad/s"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "f_out": "Hz",
                "V_out": "V (line-to-line RMS)",
                "vf_ratio": "V/Hz",
                "V_dc": "V",
                "omega_m": "rad/s",
                "speed_rpm": "rpm",
                "omega_sync": "rad/s",
                "slip": "-",
                "torque": "N.m",
                "P_mech": "W",
                "efficiency": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"f_set": 40.0, "T_load": 50.0, "duration_s": 3.0, "dt": 0.01})
    print(
        f"Final speed: {r['speed_rpm'][-1]:.1f} rpm  "
        f"slip={r['slip'][-1]:.4f}  torque={r['torque'][-1]:.1f} N.m  "
        f"V_dc={r['V_dc'][-1]:.1f} V  eta={r['efficiency'][-1]:.3f}"
    )
