"""
EC176 -- PMSM -- F2a dq-Frame Dynamic Model -- predict interface.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import PMSMF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")

def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)


class ComponentModel:
    component_id = "EC176"
    component_name = "Permanent Magnet Synchronous Motor (PMSM)"
    fidelity = "F2a -- dq-Frame Dynamic Model"
    version = "1.0.0"

    def __init__(self, params=None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = PMSMF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            mode : str  -- "direct" or "speed_control" (default "speed_control")
            speed_ref_rpm : float -- speed reference [rpm] (for speed_control mode)
            T_load_Nm : float or callable -- load torque [Nm] (default 0.0)
            v_d : float or callable -- d-axis voltage [V] (for direct mode)
            v_q : float or callable -- q-axis voltage [V] (for direct mode)
            dt : float -- time step [s] (default 1e-4)
            duration_s : float -- simulation duration [s] (default 1.0)
        """
        mode = inputs.get("mode", "speed_control")
        T_load = inputs.get("T_load_Nm", 0.0)
        dt = inputs.get("dt", 1e-4)
        duration = inputs.get("duration_s", 1.0)

        if mode == "direct":
            v_d = inputs.get("v_d", 0.0)
            v_q = inputs.get("v_q", 50.0)
            return self._model.simulate_direct(v_d, v_q, T_load, dt, duration)
        else:
            speed_ref = inputs.get("speed_ref_rpm", 1500.0)
            return self._model.simulate_speed_control(speed_ref, T_load, dt, duration)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "mode": {"type": "str", "options": ["direct", "speed_control"]},
                "speed_ref_rpm": {"unit": "rpm", "range": [0, 6000]},
                "T_load_Nm": {"unit": "Nm", "range": [0, 50]},
                "v_d": {"unit": "V", "range": [-200, 200]},
                "v_q": {"unit": "V", "range": [-200, 200]},
                "dt": {"unit": "s", "range": [1e-5, 1e-3]},
                "duration_s": {"unit": "s", "range": [0.01, 10.0]},
            },
            "outputs": {
                "t": "s", "speed_rpm": "rpm", "torque": "Nm",
                "i_d": "A", "i_q": "A", "power": "W", "omega_m": "rad/s",
            },
            "source": self._raw.get("unit", {}).get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"mode": "speed_control", "speed_ref_rpm": 1500.0, "T_load_Nm": 5.0})
    print(f"Final speed={r['speed_rpm'][-1]:.1f} rpm, "
          f"Final torque={r['torque'][-1]:.2f} Nm, "
          f"Final power={r['power'][-1]:.1f} W")
