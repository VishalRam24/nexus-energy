"""
EC141 -- Anaerobic Digester (Thermophilic) -- F2a Simplified ADM1 + Thermal
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import AnaerobicDigesterThermophilicF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the thermophilic AD F2a kinetics+thermal model."""

    component_id = "EC141"
    component_name = "Anaerobic Digester (Thermophilic)"
    fidelity = "F2a -- Simplified ADM1 Kinetics + Lumped Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = AnaerobicDigesterThermophilicF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a dynamic CSTR digestion simulation.

        inputs:
            S_in_COD : float       influent substrate [kgCOD/m3]  (default 50.0)
            Q_in : float           feed rate [m3/day]             (default 6.67 -> HRT 15 d)
            T0_degC : float        initial temperature [degC]     (default 55.0)
            duration_days : float  horizon [days]                 (default 60.0)
            dt_days : float        output step [days]             (default 0.5)
        """
        S_in = inputs.get("S_in_COD", 50.0)
        Q_in = inputs.get("Q_in", 6.667)
        T0 = inputs.get("T0_degC", 55.0)
        dur = inputs.get("duration_days", 60.0)
        dt = inputs.get("dt_days", 0.5)

        return self._model.simulate(S_in, Q_in, T0, dur, dt)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "S_in_COD": {"unit": "kgCOD/m3", "range": [0, 200]},
                "Q_in": {"unit": "m3/day", "range": [0, 50]},
                "T0_degC": {"unit": "degC", "range": [10, 70]},
                "duration_days": {"unit": "days", "range": [1, 365]},
                "dt_days": {"unit": "days", "range": [0.05, 2.0]},
            },
            "outputs": {
                "t": "days",
                "Xc/Ss/Sa_VFA/Xaci/Xmet": "kgCOD/m3",
                "temperature": "K",
                "Q_CH4_m3_day": "m3/day",
                "Q_biogas_m3_day": "m3/day",
                "energy_kWh_day": "kWh/day",
                "heating_demand_W": "W",
                "HRT_days": "days",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"S_in_COD": 50.0, "Q_in": 6.667, "duration_days": 60.0, "dt_days": 1.0})
    print(f"Final T: {r['temperature_degC'][-1]:.2f} C, "
          f"CH4: {r['Q_CH4_m3_day'][-1]:.1f} m3/day, "
          f"biogas: {r['Q_biogas_m3_day'][-1]:.1f} m3/day, "
          f"energy: {r['energy_kWh_day'][-1]:.1f} kWh/day, "
          f"heat demand: {r['heating_demand_W'][-1]/1000:.1f} kW, "
          f"HRT: {r['HRT_days']:.1f} d")
