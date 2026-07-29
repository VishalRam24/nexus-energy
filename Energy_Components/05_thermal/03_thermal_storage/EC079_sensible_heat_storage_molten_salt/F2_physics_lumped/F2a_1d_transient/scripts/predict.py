"""
EC079 -- Molten Salt TES -- F2a 1D Transient
Standardised predict() / get_info() interface.
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import MoltenSaltTES_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for Molten Salt TES F2a 1D transient model."""

    component_id = "EC079"
    component_name = "Molten Salt TES"
    fidelity = "F2a — 1D Transient Stratified Model with T-dependent Properties"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = MoltenSaltTES_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation.

        inputs:
            m_dot_charge : float [kg/s] (default 100.0)
            T_charge_in : float [K] (default 838.15 = 565 C)
            m_dot_discharge : float [kg/s] (default 0.0)
            T_discharge_in : float [K] (default 563.15 = 290 C)
            T_init : list or None
            init_mode : str ('cold', 'hot', 'linear')
            dt : float [s] (default 60.0)
            duration_s : float [s] (default 14400.0 = 4h)
        """
        m_ch = inputs.get("m_dot_charge", 100.0)
        T_ch = inputs.get("T_charge_in", 838.15)
        m_dis = inputs.get("m_dot_discharge", 0.0)
        T_dis = inputs.get("T_discharge_in", 563.15)
        T_init = inputs.get("T_init", None)
        init_mode = inputs.get("init_mode", "cold")
        dt = inputs.get("dt", 60.0)
        dur = inputs.get("duration_s", 14400.0)

        if T_init is None:
            T_init = self._model.initial_temperature_profile(init_mode)

        return self._model.simulate(m_ch, T_ch, m_dis, T_dis, T_init, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "m_dot_charge": {"unit": "kg/s", "range": [0, 500]},
                "T_charge_in": {"unit": "K", "range": [563.15, 873.15]},
                "m_dot_discharge": {"unit": "kg/s", "range": [0, 500]},
                "T_discharge_in": {"unit": "K", "range": [533.15, 573.15]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_profiles": "K (n_nodes x n_times)",
                "T_top": "K",
                "T_bottom": "K",
                "T_mean": "K",
                "E_stored_MWh": "MWh",
                "stratification_K": "K",
                "rho_mean": "kg/m3",
                "cp_mean": "J/(kg.K)",
                "k_mean": "W/(m.K)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"duration_s": 3600.0, "dt": 60.0, "m_dot_charge": 100.0})
    print(f"Final T_top: {r['T_top'][-1]:.2f} K ({r['T_top'][-1]-273.15:.1f} C)")
    print(f"Final E_stored: {r['E_stored_MWh'][-1]:.2f} MWh")
