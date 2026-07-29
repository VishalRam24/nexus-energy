"""
EC063 -- Vertical Axis Wind Turbine (VAWT) -- F2a DMST + Rotor Dynamics
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import VAWT_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for VAWT F2a DMST + rotor-dynamics model."""

    component_id = "EC063"
    component_name = "Vertical Axis Wind Turbine (VAWT)"
    fidelity = "F2a -- Double-Multiple-Streamtube aerodynamics + rotor-dynamics ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            # allow overriding nested 'turbine'/'drivetrain'/'environment' blocks
            for block, vals in params.items():
                if block in self._raw and isinstance(vals, dict):
                    self._raw[block].update(vals)
        self._model = VAWT_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a dynamic rotor simulation.

        inputs:
            wind_speed   : float (or list/callable) -- freestream wind [m/s]
            T_load_Nm    : float (or callable)       -- generator torque [N.m], default 150
            omega0       : float -- initial rotor speed [rad/s], default 8.0
            dt           : float -- output step [s], default 0.5
            duration_s   : float -- total time [s], default 120.0

        Returns a dict of time-series arrays: t, omega, rpm, tip_speed_ratio,
        cp, power_aero, power_elec, torque_aero, wind_speed, torque_load.
        """
        U = inputs.get("wind_speed", 10.0)
        T_load = inputs.get("T_load_Nm", 150.0)
        omega0 = inputs.get("omega0", 8.0)
        dt = inputs.get("dt", 0.5)
        dur = inputs.get("duration_s", 120.0)
        return self._model.simulate(U, T_load=T_load, omega0=omega0,
                                    dt=dt, duration_s=dur)

    def cp_curve(self, n=40):
        """Return (lambda, Cp) arrays of the steady DMST performance curve."""
        import numpy as np
        lam = np.linspace(0.5, 8.0, n)
        cp = np.array([self._model.cp(l) for l in lam])
        return {"tip_speed_ratio": lam, "cp": cp}

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "wind_speed": {"unit": "m/s", "range": [0.5, 30.0]},
                "T_load_Nm": {"unit": "N.m", "range": [0.0, 1500.0]},
                "omega0": {"unit": "rad/s", "range": [0.0, 60.0]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "omega": "rad/s",
                "rpm": "rev/min",
                "tip_speed_ratio": "-",
                "cp": "-",
                "power_aero": "W",
                "power_elec": "W",
                "torque_aero": "N.m",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    cpmax, lam = m._model.cp_max()
    print(f"Peak Cp = {cpmax:.3f} at TSR = {lam:.2f}")
    r = m.predict({"wind_speed": 10.0, "T_load_Nm": 250.0,
                   "duration_s": 60.0, "dt": 1.0})
    print(f"Steady: TSR={r['tip_speed_ratio'][-1]:.2f}, "
          f"Cp={r['cp'][-1]:.3f}, rpm={r['rpm'][-1]:.1f}, "
          f"P_elec={r['power_elec'][-1]:.0f} W")
