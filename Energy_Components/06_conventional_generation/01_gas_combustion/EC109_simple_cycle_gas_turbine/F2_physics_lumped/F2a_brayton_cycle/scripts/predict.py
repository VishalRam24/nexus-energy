"""
EC109 -- Simple Cycle Gas Turbine -- F2a Brayton Cycle
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import SimpleGasTurbine_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for Simple Cycle Gas Turbine F2a Brayton cycle model."""

    component_id = "EC109"
    component_name = "Simple Cycle Gas Turbine"
    fidelity = "F2a -- Brayton Cycle with Temperature-Dependent Properties"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = SimpleGasTurbine_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run Brayton cycle analysis.

        inputs:
            TIT_K : float           Turbine inlet temperature [K]
            PR : float              Compressor pressure ratio
            m_dot_air : float       Air mass flow [kg/s]
            T_amb_K : float         Ambient temperature [K]
            load_fraction : float   Part-load (0.3-1.0, uses TIT modulation)
        """
        TIT = inputs.get("TIT_K", None)
        PR = inputs.get("PR", None)
        m_dot = inputs.get("m_dot_air", None)
        T_amb = inputs.get("T_amb_K", None)
        lf = inputs.get("load_fraction", None)

        if lf is not None and lf < 1.0:
            result = self._model.part_load(lf, T_amb)
        else:
            result = self._model.brayton_cycle(TIT, PR, m_dot, T_amb)
        return result

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "TIT_K": {"unit": "K", "range": [1073.15, 1873.15]},
                "PR": {"unit": "-", "range": [5, 40]},
                "m_dot_air": {"unit": "kg/s", "range": [20, 500]},
                "T_amb_K": {"unit": "K", "range": [253.15, 323.15]},
                "load_fraction": {"unit": "-", "range": [0.3, 1.0]},
            },
            "outputs": {
                "W_elec_MW": "MW",
                "eta_electrical": "-",
                "eta_thermal": "-",
                "heat_rate_kJ_kWh": "kJ/kWh",
                "T_exhaust_K": "K",
                "SFC_kg_kWh": "kg/kWh",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({})
    print(f"W_elec: {r['W_elec_MW']:.1f} MW, eta_elec: {r['eta_electrical']:.3f}")
