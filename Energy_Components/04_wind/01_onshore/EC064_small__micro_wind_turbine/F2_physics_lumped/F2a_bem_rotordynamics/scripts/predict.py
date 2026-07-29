"""
EC064 -- Small / Micro Wind Turbine -- F2a BEM Rotor-Dynamics
Standardised predict() / get_info() interface (mirrors EC001 F2a).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import SmallWindTurbineF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC064 F2a BEM rotor-dynamics model."""

    component_id = "EC064"
    component_name = "Small / Micro Wind Turbine (HAWT)"
    fidelity = "F2a -- BEM Cp(lambda,beta) + Rotor-Dynamics ODE + PMSG load"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = SmallWindTurbineF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a rotor spin-up / transient simulation.

        inputs:
            wind_speed_ms : float (or list/callable for time-varying)
            omega0_rad_s  : float (initial rotor speed, default 1.0)
            pitch_deg     : float (blade pitch, default 0.0)
            dt            : float (default 0.05)
            duration_s    : float (default 60.0)
            TI            : float (turbulence intensity, default 0.0)
        """
        U = inputs.get("wind_speed_ms", 8.0)
        if isinstance(U, (list, tuple)):
            import numpy as np
            arr = np.asarray(U, dtype=float)
            dur = inputs.get("duration_s", 60.0)
            tg = np.linspace(0.0, dur, len(arr))
            U = (lambda a, g: (lambda t: float(np.interp(t, g, a))))(arr, tg)

        omega0 = inputs.get("omega0_rad_s", 1.0)
        beta = inputs.get("pitch_deg", 0.0)
        dt = inputs.get("dt", 0.05)
        dur = inputs.get("duration_s", 60.0)
        TI = inputs.get("TI", 0.0)

        return self._model.simulate(U, omega0=omega0, beta_deg=beta,
                                    dt=dt, duration_s=dur, TI=TI)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "wind_speed_ms": {"unit": "m/s", "range": [0, 30]},
                "omega0_rad_s": {"unit": "rad/s", "range": [0, 40]},
                "pitch_deg": {"unit": "deg", "range": [0, 30]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
                "TI": {"unit": "-", "range": [0, 0.4]},
            },
            "outputs": {
                "t": "s",
                "omega": "rad/s",
                "rpm": "rev/min",
                "tsr": "-",
                "Cp": "-",
                "P_aero": "W",
                "P_elec": "W",
                "T_aero": "N.m",
                "T_gen": "N.m",
                "T_loss": "N.m",
                "efficiency": "-",
                "wind_speed": "m/s",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"wind_speed_ms": 8.0, "duration_s": 30.0, "dt": 0.1})
    print(f"Final omega: {r['omega'][-1]:.3f} rad/s "
          f"({r['rpm'][-1]:.1f} rpm), TSR={r['tsr'][-1]:.2f}, "
          f"Cp={r['Cp'][-1]:.3f}, P_elec={r['P_elec'][-1]:.1f} W")
