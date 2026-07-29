"""
EC155 -- Geothermal District Heating -- F2a Lumped Network Thermal Transient
Standardised predict() / get_info() interface (mirrors EC001 F2a).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import GeothermalDH_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC155 F2a geothermal DH network transient."""

    component_id = "EC155"
    component_name = "Geothermal District Heating"
    fidelity = "F2a -- Lumped Network Thermal Transient (supply/return ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = GeothermalDH_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the dynamic network thermal simulation.

        inputs:
            Q_load_kW    : float or callable(t)->kW  (district heat demand)
            T_supply0    : float (degC) initial supply temperature
            T_return0    : float (degC) initial return temperature
            T_geo_source : float or callable(t)->degC  (wellhead temperature)
            boiler_on    : bool (enable peak boiler)
            dt           : float (s) output interval (default 60)
            duration_s   : float (s) horizon (default 86400 = 1 day)
        """
        return self._model.simulate(
            Q_load_kW=inputs.get("Q_load_kW", None),
            T_s0=inputs.get("T_supply0", None),
            T_r0=inputs.get("T_return0", None),
            T_geo_source=inputs.get("T_geo_source", None),
            dt=inputs.get("dt", 60.0),
            duration_s=inputs.get("duration_s", 86400.0),
            boiler_on=inputs.get("boiler_on", True),
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "Q_load_kW": {"unit": "kW", "range": [0, 20000]},
                "T_supply0": {"unit": "degC", "range": [20, 120]},
                "T_return0": {"unit": "degC", "range": [10, 90]},
                "T_geo_source": {"unit": "degC", "range": [50, 150]},
                "boiler_on": {"unit": "bool"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_supply": "degC",
                "T_return": "degC",
                "T_reinject": "degC",
                "Q_geo_kW": "kW",
                "Q_boiler_kW": "kW",
                "Q_load_kW": "kW",
                "Q_cascade_kW": "kW",
                "Q_loss_kW": "kW",
                "Q_delivered_kW": "kW",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"Q_load_kW": 6000.0, "duration_s": 43200.0, "dt": 600.0})
    print(f"Final T_supply: {r['T_supply'][-1]:.2f} C, "
          f"T_return: {r['T_return'][-1]:.2f} C, "
          f"Q_geo: {r['Q_geo_kW'][-1]:.1f} kW, "
          f"Q_boiler: {r['Q_boiler_kW'][-1]:.1f} kW, "
          f"T_reinject: {r['T_reinject'][-1]:.2f} C")
