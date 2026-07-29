"""
EC093 — Adsorption Chiller — F2a Physics-Lumped (Adsorption Kinetics)
Standardised predict() / get_info() interface (mirrors EC001 F2a).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import AdsorptionChillerF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC093 adsorption-chiller F2a kinetic model."""

    component_id = "EC093"
    component_name = "Adsorption Chiller"
    fidelity = "F2a — Physics-Lumped Adsorption Cycle (D-A isotherm + LDF kinetics + bed thermal ODEs)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = AdsorptionChillerF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the cyclic adsorption-chiller simulation.

        inputs:
            T_hot       : float [degC]  driving hot-water inlet (default 85)
            T_cool      : float [degC]  cooling-water inlet (default 30)
            T_chilled   : float [degC]  evaporator / chilled-water (default 14)
            t_half_cycle: float [s]     half-cycle switch time (default 400)
            n_cycles    : int           cycles to reach steady state (default 8)
        """
        set_temps = {}
        for k_in, k_par in [("T_hot", "T_hot"), ("T_cool", "T_cool"),
                            ("T_chilled", "T_chilled"), ("t_half_cycle", "t_half_cycle")]:
            if k_in in inputs:
                set_temps[k_par] = inputs[k_in]
        n_cycles = inputs.get("n_cycles", 8)
        return self._model.simulate(n_cycles=n_cycles, set_temps=set_temps)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_hot": {"unit": "degC", "range": [55, 95]},
                "T_cool": {"unit": "degC", "range": [22, 40]},
                "T_chilled": {"unit": "degC", "range": [5, 20]},
                "t_half_cycle": {"unit": "s", "range": [100, 1200]},
                "n_cycles": {"unit": "-", "range": [1, 60]},
            },
            "outputs": {
                "thermal_COP": "-",
                "cooling_power_kW": "kW",
                "driving_heat_mean_kW": "kW",
                "SCP_W_per_kg": "W/kg silica-gel",
                "cycle_time_s": "s",
                "T_bed_ads": "K array", "T_bed_des": "K array",
                "w_ads": "kg/kg array", "w_des": "kg/kg array",
                "Q_evap": "W array", "Q_des": "W array",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    info = m.get_info()
    print(f"{info['component_id']} {info['component_name']} — {info['fidelity']}")
    r = m.predict({"T_hot": 85.0, "T_cool": 30.0, "T_chilled": 14.0,
                   "t_half_cycle": 400.0, "n_cycles": 8})
    print(f"thermal_COP        = {r['thermal_COP']:.3f}")
    print(f"cooling_power      = {r['cooling_power_kW']:.2f} kW")
    print(f"driving_heat       = {r['driving_heat_mean_kW']:.2f} kW")
    print(f"SCP                = {r['SCP_W_per_kg']:.1f} W/kg")
    print(f"cycle_time         = {r['cycle_time_s']:.0f} s")
    print(f"dw adsorbed/desorbed = {r['dw_adsorbed']:.4f} / {r['dw_desorbed']:.4f} kg/kg")
