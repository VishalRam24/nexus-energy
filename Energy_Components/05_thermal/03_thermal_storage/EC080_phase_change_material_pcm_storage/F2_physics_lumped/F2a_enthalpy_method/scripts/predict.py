"""
EC080 -- Phase Change Material (PCM) Storage -- F2a Enthalpy Method
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import PCMStorage_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for PCM Storage F2a enthalpy method model."""

    component_id = "EC080"
    component_name = "Phase Change Material (PCM) Storage"
    fidelity = "F2a -- Enthalpy Method with Multi-Node ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = PCMStorage_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation.

        inputs:
            T_htf_K : float       HTF inlet temperature [K] (default 353.15 for charge)
            m_dot_htf : float     HTF mass flow [kg/s] (default 0.5)
            T_init_K : float      Initial PCM temperature [K] (default 293.15)
            dt : float            Time step [s] (default 10.0)
            duration_s : float    Simulation duration [s] (default 3600.0)
            mode : str            'charge' or 'discharge'
        """
        T_htf = inputs.get("T_htf_K", 353.15)
        m_dot = inputs.get("m_dot_htf", 0.5)
        T_init = inputs.get("T_init_K", 293.15)
        dt = inputs.get("dt", 10.0)
        dur = inputs.get("duration_s", 3600.0)
        mode = inputs.get("mode", "charge")

        result = self._model.simulate(T_htf, m_dot, T_init, dt, dur, mode)
        return result

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_htf_K": {"unit": "K", "range": [283.15, 373.15]},
                "m_dot_htf": {"unit": "kg/s", "range": [0.0, 5.0]},
                "T_init_K": {"unit": "K", "range": [273.15, 373.15]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
                "mode": {"values": ["charge", "discharge", "cycle"]},
            },
            "outputs": {
                "t": "s",
                "T_nodes": "K (N_nodes x Nt)",
                "T_mean": "K",
                "H_nodes": "J/kg (N_nodes x Nt)",
                "liquid_fraction": "- (N_nodes x Nt)",
                "lf_mean": "-",
                "E_stored_J": "J",
                "Q_rate_W": "W",
                "T_htf_out": "K",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"T_htf_K": 353.15, "duration_s": 600.0, "dt": 10.0})
    print(f"Final T_mean: {r['T_mean'][-1]:.2f} K, Final lf_mean: {r['lf_mean'][-1]:.3f}")
