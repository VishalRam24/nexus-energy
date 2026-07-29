"""
EC101 -- Combined Cycle Gas Turbine (CCGT) -- F2a Thermo Cycle SS
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import CCGT_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for CCGT F2a thermodynamic cycle model."""

    component_id = "EC101"
    component_name = "Combined Cycle Gas Turbine (CCGT)"
    fidelity = "F2a -- Steady-State Thermodynamic Cycle (Brayton + Rankine)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = CCGT_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run steady-state combined cycle analysis.

        inputs:
            TIT_K : float          Turbine inlet temperature [K]
            PR : float             Compressor pressure ratio
            m_dot_air : float      Air mass flow [kg/s]
            load_fraction : float  Part-load fraction (0.3-1.0)
        """
        TIT = inputs.get("TIT_K", None)
        PR = inputs.get("PR", None)
        m_dot = inputs.get("m_dot_air", None)
        lf = inputs.get("load_fraction", 1.0)

        result = self._model.combined_cycle(TIT, PR, m_dot, lf)
        return result

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "TIT_K": {"unit": "K", "range": [1273.15, 1873.15]},
                "PR": {"unit": "-", "range": [8, 35]},
                "m_dot_air": {"unit": "kg/s", "range": [100, 800]},
                "load_fraction": {"unit": "-", "range": [0.3, 1.0]},
            },
            "outputs": {
                "W_total_MW": "MW",
                "W_gt_elec_MW": "MW",
                "W_st_elec_MW": "MW",
                "eta_combined": "-",
                "heat_rate_kJ_kWh": "kJ/kWh",
                "T_exhaust_K": "K",
                "m_dot_fuel_kgs": "kg/s",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({})
    print(f"W_total: {r['W_total_MW']:.1f} MW, eta_cc: {r['eta_combined']:.3f}")
