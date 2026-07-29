"""
EC133 -- Tidal Lagoon -- F2a Physics-Lumped Water-Level ODE
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import TidalLagoonF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC133 tidal-lagoon F2a lumped ODE model."""

    component_id = "EC133"
    component_name = "Tidal Lagoon"
    fidelity = "F2a -- Physics-Lumped Lagoon Water-Level ODE (0D, two-way generation)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = TidalLagoonF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a tidal-lagoon dynamic simulation.

        inputs:
            n_cycles : int   number of tidal periods to simulate (default 2)
            H_start_hold_m : float  holding head before generation [m] (optional)
            tidal_amplitude_m : float  sea amplitude override [m] (optional)
            n_eval : int     number of output samples (default 2000)

        returns dict with time series + per-cycle energy summary.
        """
        n_cycles = int(inputs.get("n_cycles", 2))
        n_eval = int(inputs.get("n_eval", 2000))
        H_hold = inputs.get("H_start_hold_m", None)

        if "tidal_amplitude_m" in inputs:
            self._model.a = float(inputs["tidal_amplitude_m"])

        r = self._model.simulate(n_cycles=n_cycles, H_hold=H_hold, n_eval=n_eval)
        return r

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "n_cycles": {"unit": "-", "range": [1, 10]},
                "H_start_hold_m": {"unit": "m", "range": [1.0, 6.0]},
                "tidal_amplitude_m": {"unit": "m", "range": [0.5, 8.0]},
                "n_eval": {"unit": "-"},
            },
            "outputs": {
                "t": "s",
                "z_sea": "m",
                "z_lagoon": "m",
                "head": "m",
                "flow": "m3/s",
                "power_MW": "MW",
                "energy_per_cycle_MWh": "MWh",
                "avg_power_MW": "MW",
                "capacity_factor": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"n_cycles": 2})
    print(
        f"Energy/cycle: {r['energy_per_cycle_MWh']:.1f} MWh, "
        f"avg power: {r['avg_power_MW']:.1f} MW, "
        f"capacity factor: {r['capacity_factor']:.3f}, "
        f"peak |head|: {abs(r['head']).max():.2f} m"
    )
