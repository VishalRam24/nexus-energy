"""
EC127 -- Gravity Energy Storage -- F2a Hoist Dynamics
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import GravityHoistF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC127 gravity-storage F2a hoist-dynamics model."""

    component_id = "EC127"
    component_name = "Gravity Energy Storage (solid mass / tower)"
    fidelity = "F2a -- Hoist Dynamics (Newton's 2nd law ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = GravityHoistF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a hoist stroke simulation (charge=lift or discharge=lower).

        inputs:
            mode : "charge" | "discharge"   (default "charge")
            v_target : float [m/s]          (default v_max)
            dt : float [s]                  (default 1.0)
            duration_s : float [s]          (default: full stroke)
        returns dict with time-series + scalar energy summary.
        """
        mode = inputs.get("mode", "charge")
        v_target = inputs.get("v_target", None)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", None)
        x0 = inputs.get("x0", None)
        return self._model.simulate(mode=mode, v_target=v_target, x0=x0,
                                    dt=dt, duration_s=dur)

    def round_trip_efficiency(self, v_target=None, dt=2.0) -> float:
        return self._model.round_trip_efficiency(v_target=v_target, dt=dt)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "mode": {"values": ["charge", "discharge"]},
                "v_target": {"unit": "m/s", "range": [0.0, self._model.v_max]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "x": "m",
                "v": "m/s",
                "height": "m",
                "soc": "-",
                "F_cable": "N",
                "P_mech": "W",
                "P_elec": "W (neg=charge draw, pos=discharge return)",
                "E_stored_kwh": "kWh",
                "E_elec_kwh": "kWh",
            },
            "capacity_kwh": self._model.energy_capacity_kwh(),
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"mode": "charge", "v_target": 3.0, "dt": 5.0})
    print(f"Charge: reached x={r['x'][-1]:.1f} m, SOC={r['soc'][-1]:.3f}, "
          f"E_elec_in={abs(r['E_elec_kwh']):.1f} kWh in {r['t'][-1]:.0f} s")
    eta = m.round_trip_efficiency(v_target=3.0, dt=5.0)
    print(f"Round-trip efficiency = {eta*100:.1f} %")
