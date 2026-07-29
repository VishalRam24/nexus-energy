"""
EC033 -- Iron-Air Battery (Fe-Air) -- F2a Physics-Lumped Electrochemical
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import IronAirF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC033 Iron-Air F2a physics-lumped model."""

    component_id = "EC033"
    component_name = "Iron-Air Battery (Fe-Air)"
    fidelity = "F2a -- Physics-Lumped Electrochemical Model with Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = IronAirF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation.

        inputs:
            current_density_A_cm2 : float (or callable) [+ = discharge, - = charge]
            T_init_K   : float (initial temperature, default 298.15)
            dt         : float (default 10.0 s)
            duration_s : float (default 3600.0 s)
            soc_init   : float (default 0.5)
        """
        j = inputs.get("current_density_A_cm2", 0.02)
        T0 = inputs.get("T_init_K", 298.15)
        dt = inputs.get("dt", 10.0)
        dur = inputs.get("duration_s", 3600.0)
        soc0 = inputs.get("soc_init", 0.5)

        return self._model.simulate(j, T0, dt, dur, soc_init=soc0)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_density_A_cm2": {"unit": "A/cm2", "range": [-0.06, 0.06],
                                          "note": "+discharge / -charge"},
                "T_init_K": {"unit": "K", "range": [273.15, 333.15]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
                "soc_init": {"unit": "-", "range": [0.0, 1.0]},
            },
            "outputs": {
                "t": "s",
                "voltage": "V",
                "ocv": "V",
                "power_density": "W/cm2",
                "soc": "-",
                "temperature": "K",
                "coulombic_eff": "-",
                "her_current": "A/cm2",
                "overpotentials": "dict of V arrays",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    # Discharge sample
    rd = m.predict({"current_density_A_cm2": 0.02, "duration_s": 600.0, "dt": 60.0})
    # Charge sample
    rc = m.predict({"current_density_A_cm2": -0.02, "duration_s": 600.0, "dt": 60.0})
    eta_rt = m._model.round_trip_efficiency(0.02, 298.15)
    print(f"Discharge V = {rd['voltage'][0]:.4f} V | OCV = {rd['ocv'][0]:.4f} V "
          f"| Charge V = {rc['voltage'][0]:.4f} V")
    print(f"Coulombic eff (charge) = {rc['coulombic_eff'][0]:.4f} | "
          f"Round-trip eff = {eta_rt:.4f}")
    print(f"Discharge final T = {rd['temperature'][-1]:.2f} K, SOC = {rd['soc'][-1]:.3f}")
