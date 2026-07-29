"""
EC152 -- Flash Steam Geothermal Plant -- F2a Flash Thermodynamics
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import FlashSteamGeothermalF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC152 flash-steam geothermal F2a model."""

    component_id = "EC152"
    component_name = "Flash Steam Geothermal Plant"
    fidelity = "F2a -- Flash Thermodynamics with Lumped Separator/Turbine ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = FlashSteamGeothermalF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic flash-steam plant simulation.

        inputs:
            m_dot_brine_kgs : float (or callable t->kg/s)  default 100.0
            T_geo_C   : float  brine wellhead temperature (default 240.0)
            T_reject_C: float  cooling rejection temperature (default 50.0)
            T_flash_C : float  separator T (default: optimal geometric mean)
            dt        : float  (default 0.5)
            duration_s: float  (default 120.0)
        """
        m_dot = inputs.get("m_dot_brine_kgs", self._model.m_dot_design)
        T_geo = inputs.get("T_geo_C", self._model.T_geo_design)
        T_rej = inputs.get("T_reject_C", self._model.T_reject_design)
        T_flash = inputs.get("T_flash_C", None)
        dt = inputs.get("dt", 0.5)
        dur = inputs.get("duration_s", 120.0)

        return self._model.simulate(m_dot, T_geo, T_rej, dt, dur, T_flash_c=T_flash)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "m_dot_brine_kgs": {"unit": "kg/s", "range": [10, 500]},
                "T_geo_C": {"unit": "degC", "range": [180, 320]},
                "T_reject_C": {"unit": "degC", "range": [10, 60]},
                "T_flash_C": {"unit": "degC", "range": [100, 250]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "net_power_kW": "kW",
                "steam_flow_kgs": "kg/s",
                "brine_flow_kgs": "kg/s",
                "steam_fraction": "-",
                "eta_utilization": "-",
                "eta_carnot": "-",
                "specific_work_kJkg": "kJ/kg",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"m_dot_brine_kgs": 100.0, "duration_s": 60.0, "dt": 1.0})
    print(f"Steady P={r['P_steady_kW']:.1f} kW | final P={r['net_power_kW'][-1]:.1f} kW | "
          f"x={r['steam_fraction'][-1]:.3f} | eta_util={r['eta_utilization'][-1]:.3f} | "
          f"eta_carnot={r['eta_carnot'][-1]:.3f} | T_flash={r['T_flash_C']:.1f} C")
