"""
EC078 -- Hot Water Tank TES -- F2a 1D Transient
Standardised predict() / get_info() interface.
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import HotWaterTank_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for Hot Water Tank F2a 1D transient model."""

    component_id = "EC078"
    component_name = "Hot Water Tank TES"
    fidelity = "F2a — 1D Transient Stratified Tank (20-node)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = HotWaterTank_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation.

        inputs:
            m_dot_charge : float [kg/s] (default 0.5)
            T_charge_in : float [K] (default 353.15)
            m_dot_discharge : float [kg/s] (default 0.0)
            T_discharge_in : float [K] (default 288.15)
            T_init : list or None (default None -> linear profile)
            dt : float [s] (default 10.0)
            duration_s : float [s] (default 3600.0)
        """
        m_ch = inputs.get("m_dot_charge", 0.5)
        T_ch = inputs.get("T_charge_in", 353.15)
        m_dis = inputs.get("m_dot_discharge", 0.0)
        T_dis = inputs.get("T_discharge_in", 288.15)
        T_init = inputs.get("T_init", None)
        dt = inputs.get("dt", 10.0)
        dur = inputs.get("duration_s", 3600.0)

        return self._model.simulate(m_ch, T_ch, m_dis, T_dis, T_init, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "m_dot_charge": {"unit": "kg/s", "range": [0, 2.0]},
                "T_charge_in": {"unit": "K", "range": [313.15, 373.15]},
                "m_dot_discharge": {"unit": "kg/s", "range": [0, 2.0]},
                "T_discharge_in": {"unit": "K", "range": [278.15, 313.15]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_profiles": "K (n_nodes x n_times)",
                "T_top": "K",
                "T_bottom": "K",
                "T_mean": "K",
                "E_stored_J": "J",
                "E_stored_kWh": "kWh",
                "stratification_K": "K",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"duration_s": 600.0, "dt": 10.0, "m_dot_charge": 0.5})
    print(f"Final T_top: {r['T_top'][-1]:.2f} K, T_bottom: {r['T_bottom'][-1]:.2f} K, E: {r['E_stored_kWh'][-1]:.2f} kWh")
