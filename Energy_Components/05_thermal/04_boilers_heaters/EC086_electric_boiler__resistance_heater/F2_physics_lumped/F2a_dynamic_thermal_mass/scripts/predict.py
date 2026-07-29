"""
EC086 -- Electric Boiler / Resistance Heater -- F2a Dynamic Thermal Mass
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import ElectricBoilerF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC086 F2a dynamic thermal-mass boiler."""

    component_id = "EC086"
    component_name = "Electric Boiler / Resistance Heater"
    fidelity = "F2a -- Dynamic Thermal-Mass (Lumped) with Thermostat Control"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = ElectricBoilerF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a dynamic boiler simulation.

        inputs:
            T_init_K   : float    initial water temperature (default 293.15)
            mdot_kg_s  : float    load mass flow (default 0.05)
            dt         : float    output step [s] (default 5.0)
            duration_s : float    total time [s] (default 3600.0)
            T_set_K    : float    thermostat setpoint (default param)
            control    : str      'onoff' | 'modulating' (default 'onoff')
            P_input_W  : float    optional forced electrical input [W]
        """
        T0 = inputs.get("T_init_K", 293.15)
        mdot = inputs.get("mdot_kg_s", 0.05)
        dt = inputs.get("dt", 5.0)
        dur = inputs.get("duration_s", 3600.0)
        T_set = inputs.get("T_set_K", None)
        control = inputs.get("control", "onoff")
        P_in = inputs.get("P_input_W", None)

        return self._model.simulate(
            T0, mdot, dt=dt, duration_s=dur,
            T_set=T_set, control=control, P_input=P_in,
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_init_K": {"unit": "K", "range": [274.0, 372.0]},
                "mdot_kg_s": {"unit": "kg/s", "range": [0.0, 1.0]},
                "dt": {"unit": "s", "range": [0.1, 10.0]},
                "duration_s": {"unit": "s", "range": [1.0, 86400.0]},
                "T_set_K": {"unit": "K", "range": [293.0, 368.0]},
                "control": {"unit": "-", "options": ["onoff", "modulating"]},
                "P_input_W": {"unit": "W", "note": "optional forced input"},
            },
            "outputs": {
                "t": "s",
                "temperature": "K",
                "firing_fraction": "-",
                "P_elec_W": "W",
                "Q_elec_W": "W",
                "Q_loss_W": "W",
                "Q_load_W": "W",
                "efficiency": "-",
                "energy": "dict of J (conservation bookkeeping)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"T_init_K": 293.15, "mdot_kg_s": 0.05,
                   "duration_s": 3600.0, "dt": 5.0, "control": "onoff"})
    print(f"Final T: {r['temperature'][-1]-273.15:.2f} C, "
          f"E_elec: {r['energy']['E_elec_J']/3.6e6:.3f} kWh, "
          f"E_residual: {r['energy']['E_residual_J']:.2e} J")
