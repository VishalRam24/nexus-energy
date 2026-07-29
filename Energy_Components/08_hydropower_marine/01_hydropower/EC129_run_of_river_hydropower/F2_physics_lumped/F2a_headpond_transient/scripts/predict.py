"""
EC129 -- Run-of-River Hydropower -- F2a Physics-Lumped Headpond Transient
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import RunOfRiverF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the RoR F2a headpond-transient model."""

    component_id = "EC129"
    component_name = "Run-of-River Hydropower"
    fidelity = "F2a -- Physics-Lumped Headpond Transient (Darcy-Weisbach + hill chart)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = RunOfRiverF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a forebay-level transient simulation and derive power.

        inputs:
            Q_inflow_m3s : float or callable(t)  river inflow (default 50.0)
            Q_demand_m3s : float or callable(t)  turbine-flow demand (default = Q_inflow)
            z0_m         : float  initial forebay level [m] (default = z0 param)
            dt           : float  output step [s] (default 10.0)
            duration_s   : float  horizon [s] (default 3600.0)
        """
        Q_in = inputs.get("Q_inflow_m3s", 50.0)
        Q_dem = inputs.get("Q_demand_m3s", Q_in)
        z0 = inputs.get("z0_m", None)
        dt = inputs.get("dt", 10.0)
        dur = inputs.get("duration_s", 3600.0)

        return self._model.simulate(Q_in, Q_dem, z0=z0, dt=dt, duration_s=dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "Q_inflow_m3s": {"unit": "m3/s", "range": [0, 80]},
                "Q_demand_m3s": {"unit": "m3/s", "range": [0, 60]},
                "z0_m": {"unit": "m", "range": [4.0, 9.5]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "z": "m (forebay level)",
                "H_gross": "m",
                "H_net": "m",
                "head_loss": "m",
                "Q_inflow": "m3/s",
                "Q_turbine": "m3/s",
                "Q_spill": "m3/s",
                "eta": "-",
                "power_kw": "kW",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"Q_inflow_m3s": 50.0, "duration_s": 3600.0, "dt": 60.0})
    print(
        f"Final level z={r['z'][-1]:.3f} m, H_net={r['H_net'][-1]:.3f} m, "
        f"eta={r['eta'][-1]:.3f}, P={r['power_kw'][-1]:.1f} kW"
    )
