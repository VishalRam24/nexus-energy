"""
EC088 -- Oil-Fired Boiler -- F2a Dynamic Thermal Mass
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import OilBoilerF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the oil-fired boiler F2a dynamic model."""

    component_id = "EC088"
    component_name = "Oil-Fired Boiler"
    fidelity = "F2a -- Physics-Lumped Dynamic Thermal Mass with Combustion & Stack Loss"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = OilBoilerF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic boiler simulation.

        inputs:
            firing_rate   : float in [0,1] or callable f(t)->[0,1]
            T_water_init  : float degC (default 40.0)
            T_return      : float degC, heating-circuit return temp (default 60.0)
            T_ambient     : float degC (default 20.0)
            excess_air    : float lambda (default from parameters, 1.20)
            dt            : float s (default 5.0)
            duration_s    : float s (default 1800.0)
        """
        fr = inputs.get("firing_rate", 0.8)
        T0 = inputs.get("T_water_init", 40.0)
        Tr = inputs.get("T_return", 60.0)
        Ta = inputs.get("T_ambient", 20.0)
        lam = inputs.get("excess_air", None)
        dt = inputs.get("dt", 5.0)
        dur = inputs.get("duration_s", 1800.0)
        return self._model.simulate(fr, T0, Tr, Ta, lam, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "firing_rate": {"unit": "-", "range": [0.0, 1.0]},
                "T_water_init": {"unit": "degC", "range": [5.0, 95.0]},
                "T_return": {"unit": "degC", "range": [5.0, 90.0]},
                "T_ambient": {"unit": "degC", "range": [-20.0, 40.0]},
                "excess_air": {"unit": "-", "range": [1.0, 1.6]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_water_C": "degC",
                "Q_fuel_W": "W",
                "Q_useful_W": "W",
                "Q_load_W": "W",
                "Q_sensible_loss_W": "W",
                "Q_latent_loss_W": "W",
                "Q_standby_loss_W": "W",
                "T_flue_C": "degC",
                "eta_combustion": "-",
                "eta_overall": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"firing_rate": 0.8, "T_water_init": 30.0,
                   "duration_s": 1200.0, "dt": 10.0})
    print(f"Final T_water: {r['T_water_C'][-1]:.2f} C, "
          f"eta_comb: {r['eta_combustion'][-1]:.4f}, "
          f"eta_overall: {r['eta_overall'][-1]:.4f}")
