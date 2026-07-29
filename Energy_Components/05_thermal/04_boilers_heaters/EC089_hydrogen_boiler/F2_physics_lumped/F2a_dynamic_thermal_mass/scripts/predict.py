"""
EC089 -- Hydrogen Boiler -- F2a Physics-Lumped
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import HydrogenBoilerF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC089 hydrogen boiler F2a physics model."""

    component_id = "EC089"
    component_name = "Hydrogen Boiler (100% H2 Combustion)"
    fidelity = "F2a -- Physics-Lumped: combustion + flue balance + thermal-mass ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = HydrogenBoilerF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a dynamic boiler simulation.

        inputs:
            firing_rate : float in [0,1] (or callable(t))   default 1.0
            T_water_K   : float, initial water temperature   default 313.15
            dt          : float [s]                          default 1.0
            duration_s  : float [s]                          default 600.0
        """
        phi = inputs.get("firing_rate", 1.0)
        T0 = inputs.get("T_water_K", 313.15)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 600.0)
        return self._model.simulate(phi, T0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "firing_rate": {"unit": "-", "range": [0.0, 1.0]},
                "T_water_K": {"unit": "K", "range": [280.0, 373.15]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "temperature": "K (water node)",
                "firing_rate": "-",
                "heat_to_water_W": "W",
                "stack_loss_W": "W",
                "latent_recovery_W": "W",
                "efficiency": "- (HHV basis)",
                "efficiency_lhv": "-",
                "h2_flow_kg_s": "kg/s",
                "flue_temp_K": "K",
                "T_adiabatic_flame_K": "K",
                "nox_index": "- (relative thermal-NOx propensity)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    info = m.get_info()
    print(f"{info['component_id']} {info['component_name']} -- {info['fidelity']}")
    r = m.predict({"firing_rate": 1.0, "T_water_K": 300.0, "dt": 5.0, "duration_s": 600.0})
    print(f"Final water T: {r['temperature'][-1]:.2f} K")
    print(f"HHV efficiency: {r['efficiency'][-1]:.4f}  (LHV basis: {r['efficiency_lhv'][-1]:.4f})")
    print(f"Adiabatic flame T: {r['T_adiabatic_flame_K']:.0f} K, NOx index: {r['nox_index']:.3f}")
