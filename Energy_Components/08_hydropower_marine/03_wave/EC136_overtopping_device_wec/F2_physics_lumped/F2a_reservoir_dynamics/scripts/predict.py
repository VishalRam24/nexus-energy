"""
EC136 -- Overtopping Device WEC (Wave Dragon) -- F2a Physics-Lumped Reservoir Dynamics
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import OvertoppingWEC_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC136 overtopping-WEC F2a reservoir-dynamics model."""

    component_id = "EC136"
    component_name = "Overtopping Device WEC (Wave Dragon)"
    fidelity = "F2a -- Physics-Lumped Reservoir Dynamics (Van der Meer overtopping + Kaplan turbine ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = OvertoppingWEC_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the lumped reservoir-dynamics simulation.

        inputs:
            Hs_m       : float significant wave height [m] (default 3.0)
            Tz_s       : float mean wave period [s] (default 7.0)
            level0_m   : float initial reservoir level above crest [m] (default 0.5)
            dt         : float output step [s] (default 5.0)
            duration_s : float record length [s] (default 1800.0)

        returns: dict with time series and scalar summaries (mean power,
                 overall efficiency, mass-balance residual, ...).
        """
        Hs = inputs.get("Hs_m", 3.0)
        Tz = inputs.get("Tz_s", 7.0)
        level0 = inputs.get("level0_m", 0.5)
        dt = inputs.get("dt", 5.0)
        dur = inputs.get("duration_s", 1800.0)
        return self._model.simulate(Hs, Tz, level0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "Hs_m": {"unit": "m", "range": [0.5, 7.0]},
                "Tz_s": {"unit": "s", "range": [4.0, 12.0]},
                "level0_m": {"unit": "m", "range": [0.0, 3.0]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "level": "m (above crest)",
                "Q_in": "m3/s (overtopping inflow)",
                "Q_out": "m3/s (turbine discharge)",
                "head": "m",
                "power_elec_W": "W",
                "P_mean_kW": "kW",
                "eta_overall": "-",
                "mass_residual_m3": "m3",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"Hs_m": 3.0, "Tz_s": 7.0, "duration_s": 1200.0, "dt": 10.0})
    print(
        f"Mean power: {r['P_mean_kW']:.1f} kW, "
        f"overall efficiency: {r['eta_overall']*100:.1f} %, "
        f"final level: {r['level'][-1]:.3f} m, "
        f"mass residual: {r['mass_residual_m3']:.3f} m3"
    )
