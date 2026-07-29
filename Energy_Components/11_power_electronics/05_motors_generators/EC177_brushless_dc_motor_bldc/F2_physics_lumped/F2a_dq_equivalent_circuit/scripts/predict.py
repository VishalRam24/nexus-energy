"""
EC177 -- Brushless DC Motor (BLDC) -- F2a dq / phase-domain
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import BLDC_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for BLDC F2a six-step trapezoidal-EMF model."""

    component_id = "EC177"
    component_name = "Brushless DC Motor (BLDC)"
    fidelity = "F2a -- dq/phase-domain electrical + mechanical ODE (six-step)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = BLDC_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a coupled electrical+mechanical transient.

        inputs:
            T_load_Nm  : float  load torque (default 0.1)
            Vdc        : float  bus voltage (default from params)
            duty       : float  PWM duty 0..1 (default 1.0)
            dt         : float  output step s (default 2e-4)
            duration_s : float  sim time s (default 0.4)
        """
        T_load = inputs.get("T_load_Nm", 0.1)
        Vdc = inputs.get("Vdc", None)
        duty = inputs.get("duty", 1.0)
        dt = inputs.get("dt", 2e-4)
        dur = inputs.get("duration_s", 0.4)
        return self._model.simulate(T_load=T_load, Vdc=Vdc, duty=duty,
                                    dt=dt, duration_s=dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_load_Nm": {"unit": "Nm", "range": [0.0, 4.0]},
                "Vdc": {"unit": "V", "range": [12.0, 60.0]},
                "duty": {"unit": "-", "range": [0.0, 1.0]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "current": "A",
                "omega": "rad/s",
                "speed_rpm": "rpm",
                "back_emf": "V",
                "torque_e": "Nm",
                "P_mech": "W",
                "P_elec": "W",
                "efficiency": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"T_load_Nm": 0.5, "duration_s": 0.4, "dt": 2e-4})
    print(f"Final speed: {r['speed_rpm'][-1]:.1f} rpm, "
          f"Final current: {r['current_final']:.3f} A, "
          f"Final T_e: {r['torque_e_final']:.4f} Nm, "
          f"Efficiency: {r['efficiency']:.3f}")
