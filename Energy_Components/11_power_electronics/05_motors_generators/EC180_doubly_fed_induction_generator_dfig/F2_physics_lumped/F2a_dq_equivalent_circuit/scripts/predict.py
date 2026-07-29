"""
EC180 -- Doubly-Fed Induction Generator (DFIG) -- F2a dq-Frame Model
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import DFIG_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the DFIG F2a dq-frame dynamic model."""

    component_id = "EC180"
    component_name = "Doubly-Fed Induction Generator (DFIG)"
    fidelity = "F2a -- dq-Frame Doubly-Fed Induction Machine with Rotor-Side Converter Control"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = DFIG_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a DFIG dynamic simulation.

        inputs:
            mode : str -- "power_control" (default), "direct", or "mechanical"
            slip : float -- operating slip (default -0.2, super-synchronous)
            # power_control mode:
            P_stator_ref_W : float -- stator active power set-point [W] (<0 to grid)
            Q_stator_ref_VAr : float -- stator reactive power set-point [VAr]
            # direct mode:
            v_dr, v_qr : float -- rotor-side converter dq voltages [V]
            # mechanical mode:
            T_mech_Nm : float -- prime-mover torque [Nm]
            # common:
            dt : float (default 1e-4)
            duration_s : float (default 1.0)
        """
        mode = inputs.get("mode", "power_control")
        slip = inputs.get("slip", -0.2)
        dt = inputs.get("dt", 1e-4)
        dur = inputs.get("duration_s", 1.0)

        if mode == "direct":
            v_dr = inputs.get("v_dr", 0.0)
            v_qr = inputs.get("v_qr", 0.0)
            return self._model.simulate(v_dr, v_qr, slip, dt, dur)
        elif mode == "mechanical":
            v_dr = inputs.get("v_dr", 0.0)
            v_qr = inputs.get("v_qr", 50.0)
            T_mech = inputs.get("T_mech_Nm", 0.0)
            return self._model.simulate_mechanical(v_dr, v_qr, T_mech, dt, dur)
        else:  # power_control
            P_ref = inputs.get("P_stator_ref_W", -1.5e6)
            Q_ref = inputs.get("Q_stator_ref_VAr", 0.0)
            return self._model.simulate_power_control(P_ref, Q_ref, slip, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "mode": {"type": "str", "options": ["power_control", "direct", "mechanical"]},
                "slip": {"unit": "-", "range": [-0.30, 0.30]},
                "P_stator_ref_W": {"unit": "W", "range": [-2.2e6, 0.0]},
                "Q_stator_ref_VAr": {"unit": "VAr", "range": [-1.0e6, 1.0e6]},
                "v_dr": {"unit": "V", "range": [-400, 400]},
                "v_qr": {"unit": "V", "range": [-400, 400]},
                "T_mech_Nm": {"unit": "Nm"},
                "dt": {"unit": "s", "range": [1e-5, 1e-3]},
                "duration_s": {"unit": "s", "range": [0.01, 10.0]},
            },
            "outputs": {
                "t": "s",
                "speed_rpm": "rpm",
                "slip": "-",
                "torque": "Nm",
                "i_stator": "A (peak)",
                "i_rotor": "A (peak)",
                "P_stator": "W (generator: <0 to grid)",
                "Q_stator": "VAr",
                "P_rotor": "W (slip power via converter)",
                "P_grid": "W (stator + converter)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"mode": "power_control", "P_stator_ref_W": -1.5e6,
                   "Q_stator_ref_VAr": 0.0, "slip": -0.2,
                   "duration_s": 0.5, "dt": 1e-4})
    print(f"Final P_stator={r['P_stator'][-1]/1e6:.3f} MW, "
          f"Q_stator={r['Q_stator'][-1]/1e3:.1f} kVAr, "
          f"P_rotor={r['P_rotor'][-1]/1e6:.3f} MW, "
          f"slip={r['slip'][-1]:.3f}, speed={r['speed_rpm'][-1]:.0f} rpm")
