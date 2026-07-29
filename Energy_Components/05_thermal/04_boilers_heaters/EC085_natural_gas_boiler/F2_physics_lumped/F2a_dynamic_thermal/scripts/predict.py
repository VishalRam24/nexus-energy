"""
EC085 -- Natural Gas Boiler -- F2a Dynamic Thermal
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import NatGasBoiler_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for Natural Gas Boiler F2a dynamic thermal model."""

    component_id = "EC085"
    component_name = "Natural Gas Boiler"
    fidelity = "F2a -- Dynamic Thermal Mass ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = NatGasBoiler_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation.

        inputs:
            T_init_K : float       Initial boiler temperature [K]
            T_in_K : float         Return water temperature [K]
            m_dot : float          Water flow rate [kg/s]
            T_set_K : float        Setpoint temperature [K]
            dt : float             Time step [s]
            duration_s : float     Simulation duration [s]
        """
        T_init = inputs.get("T_init_K", 293.15)
        T_in = inputs.get("T_in_K", 333.15)
        m_dot = inputs.get("m_dot", 6.0)
        T_set = inputs.get("T_set_K", 353.15)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 600.0)

        result = self._model.simulate(T_init, T_in, m_dot, T_set, dt, dur)
        return result

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_init_K": {"unit": "K", "range": [273.15, 373.15]},
                "T_in_K": {"unit": "K", "range": [283.15, 363.15]},
                "m_dot": {"unit": "kg/s", "range": [0, 12]},
                "T_set_K": {"unit": "K", "range": [313.15, 373.15]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_boiler": "K",
                "T_out_water": "K",
                "modulation": "-",
                "Q_burner_W": "W",
                "Q_output_W": "W",
                "Q_loss_W": "W",
                "thermal_efficiency": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"T_init_K": 293.15, "duration_s": 120.0, "dt": 1.0})
    print(f"Final T_boiler: {r['T_boiler'][-1]:.2f} K")
