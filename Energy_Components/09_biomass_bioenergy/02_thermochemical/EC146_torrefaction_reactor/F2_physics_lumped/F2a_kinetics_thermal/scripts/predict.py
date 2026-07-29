"""
EC146 -- Torrefaction Reactor -- F2a Two-Step Kinetics + Reactor ODE
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import TorrefactionF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC146 torrefaction F2a kinetics + thermal ODE."""

    component_id = "EC146"
    component_name = "Torrefaction Reactor"
    fidelity = "F2a -- Two-Step Arrhenius Kinetics with Reactor Energy-Balance ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = TorrefactionF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic torrefaction simulation.

        inputs:
            T_set_degC : float        reactor wall / heating-medium temperature
            residence_time_min : float total reaction (residence) time
            T0_degC : float           initial solid temperature (default 25)
            dt_s : float              output sampling interval (default 5)

        Returns time-series arrays plus final-state metrics:
            mass_yield, energy_yield, hhv_upgrade, conversion, LHV_solid, ...
        """
        T_set = inputs.get("T_set_degC", 280.0)
        t_res = inputs.get("residence_time_min", 30.0)
        T0 = inputs.get("T0_degC", 25.0)
        dt = inputs.get("dt_s", 5.0)

        return self._model.simulate(T_set_degC=T_set, residence_time_min=t_res,
                                    T0_degC=T0, dt_s=dt)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_set_degC": {"unit": "degC", "range": [200, 320]},
                "residence_time_min": {"unit": "min", "range": [1, 120]},
                "T0_degC": {"unit": "degC", "range": [15, 320]},
                "dt_s": {"unit": "s", "range": [0.5, 30]},
            },
            "outputs": {
                "t": "min",
                "solid_mass": "kg/kg_feed",
                "volatiles_mass": "kg/kg_feed",
                "mass_yield": "-",
                "energy_yield": "-",
                "LHV_solid": "MJ/kg",
                "hhv_upgrade": "-",
                "conversion": "-",
                "temperature_degC": "degC",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"T_set_degC": 280.0, "residence_time_min": 30.0})
    print(f"mass_yield={r['mass_yield_final']:.3f}  "
          f"energy_yield={r['energy_yield_final']:.3f}  "
          f"HHV_upgrade={r['hhv_upgrade_final']:.3f}  "
          f"LHV_solid={r['LHV_solid_final']:.2f} MJ/kg  "
          f"T_final={r['temperature_final_degC']:.1f} degC")
