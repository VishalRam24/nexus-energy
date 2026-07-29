"""
EC151 -- Dry Steam Geothermal Plant -- F2a Physics-Lumped Steam-Turbine Expansion
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import DrySteamGeothermalF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC151 dry-steam geothermal F2a model."""

    component_id = "EC151"
    component_name = "Dry Steam Geothermal Plant"
    fidelity = "F2a -- Physics-Lumped Steam-Turbine Expansion with Wellhead Transient ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = DrySteamGeothermalF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run lumped wellhead/turbine transient simulation.

        inputs:
            P_wh_MPa       : float or callable(t) (default design point)
            P_cond_MPa     : float (default design)
            T_superheat_K  : float (default 0 = dry saturated)
            m_dot0_kgs     : float (initial flow; default quasi-steady)
            dt             : float (default 1.0 s)
            duration_s     : float (default 300.0 s)
        """
        P_wh = inputs.get("P_wh_MPa", self._model.P_wh)
        P_cond = inputs.get("P_cond_MPa", self._model.P_cond)
        T_sh = inputs.get("T_superheat_K", self._model.T_superheat)
        m0 = inputs.get("m_dot0_kgs", None)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 300.0)

        return self._model.simulate(P_wh, P_cond, T_sh, m0, dt, dur)

    def predict_steady(self, inputs: dict) -> dict:
        """Quasi-steady power/efficiency at a given flow (no integration)."""
        m_dot = inputs.get("m_dot_steam_kgs", self._model.m_dot_design)
        P_wh = inputs.get("P_wh_MPa", self._model.P_wh)
        P_cond = inputs.get("P_cond_MPa", self._model.P_cond)
        T_sh = inputs.get("T_superheat_K", self._model.T_superheat)
        x_ncg = inputs.get("ncg_mass_fraction", self._model.x_ncg)
        return self._model.power(m_dot, P_wh, P_cond, T_sh, x_ncg)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "P_wh_MPa": {"unit": "MPa", "range": [0.3, 1.5]},
                "P_cond_MPa": {"unit": "MPa", "range": [0.005, 0.05]},
                "T_superheat_K": {"unit": "K", "range": [0.0, 80.0]},
                "m_dot_steam_kgs": {"unit": "kg/s", "range": [5.0, 200.0]},
                "ncg_mass_fraction": {"unit": "-", "range": [0.0, 0.1]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "m_dot": "kg/s",
                "T_casing": "K",
                "P_net_kW": "kW",
                "P_gross_kW": "kW",
                "eta_utilization": "-",
                "eta_carnot": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    ss = m.predict_steady({})
    print(f"\nSteady design point:")
    print(f"  P_net   = {ss['P_net_kW']:.1f} kW")
    print(f"  P_gross = {ss['P_gross_kW']:.1f} kW")
    print(f"  w_spec  = {ss['w_specific_kJ_kg']:.1f} kJ/kg")
    print(f"  eta_util= {ss['eta_utilization']:.4f}")
    print(f"  eta_carnot={ss['eta_carnot']:.4f}  eta_2nd={ss['eta_2nd_law']:.4f}")
    r = m.predict({"P_wh_MPa": 0.8, "duration_s": 60.0, "dt": 2.0})
    print(f"\nTransient (60 s): m_dot {r['m_dot'][0]:.2f} -> {r['m_dot'][-1]:.2f} kg/s, "
          f"P_net {r['P_net_kW'][-1]:.1f} kW")
