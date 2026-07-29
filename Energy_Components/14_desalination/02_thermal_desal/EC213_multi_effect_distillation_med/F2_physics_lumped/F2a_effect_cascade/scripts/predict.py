"""
EC213 -- Multi-Effect Distillation (MED) -- F2a Physics-Lumped Effect Cascade
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import MED_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the MED F2a physics-lumped effect-cascade model."""

    component_id = "EC213"
    component_name = "Multi-Effect Distillation (MED)"
    fidelity = "F2a -- Physics-Lumped Effect Cascade with Transient ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = MED_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Steady-state stage-by-stage MED solve (default) or transient simulation.

        inputs (all optional, fall back to parameters.json defaults):
            N_effects   : int    number of effects
            T_top_C     : float  top brine temperature [degC]
            T_last_C    : float  last effect temperature [degC]
            T_steam_C   : float  motive steam temperature [degC]
            M_feed_kg_s : float  total seawater feed [kg/s]
            X_feed_ppm  : float  feed salinity [ppm]
            transient   : bool   if True, also run the start-up ODE
            T0_C        : float  initial effect temperature for transient [degC]
            dt          : float  output step [s] (transient)
            duration_s  : float  total time [s] (transient)
        """
        N = inputs.get("N_effects")
        T_top = inputs.get("T_top_C")
        T_last = inputs.get("T_last_C")
        T_steam = inputs.get("T_steam_C")
        M_feed = inputs.get("M_feed_kg_s")
        X_feed = inputs.get("X_feed_ppm")

        out = self._model.steady_state(
            N=N, T_top=T_top, T_last=T_last,
            M_feed=M_feed, X_feed=X_feed, T_steam=T_steam,
        )

        if inputs.get("transient", False):
            trans = self._model.simulate(
                T0_C=inputs.get("T0_C", None),
                T_steam=T_steam,
                dt=inputs.get("dt", 10.0),
                duration_s=inputs.get("duration_s", 3600.0),
            )
            out["transient"] = trans

        return out

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "N_effects": {"unit": "-", "range": [2, 16]},
                "T_top_C": {"unit": "degC", "range": [55, 85]},
                "T_last_C": {"unit": "degC", "range": [35, 70]},
                "T_steam_C": {"unit": "degC", "range": [60, 100]},
                "M_feed_kg_s": {"unit": "kg/s", "range": [20, 600]},
                "X_feed_ppm": {"unit": "ppm", "range": [30000, 50000]},
                "transient": {"unit": "bool"},
                "T0_C": {"unit": "degC"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "T_effect": "degC array (cascade)",
                "D_effect": "kg/s array (distillate per effect)",
                "B_effect": "kg/s array (brine per effect)",
                "X_brine": "ppm array",
                "Q_effect": "kW array",
                "distillate_total_kg_s": "kg/s",
                "distillate_total_m3_h": "m3/h",
                "steam_flow_kg_s": "kg/s",
                "GOR": "-",
                "recovery": "-",
                "specific_thermal_kJ_kg": "kJ/kg",
                "transient": "dict of time-series (optional)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"N_effects": 12, "transient": True, "duration_s": 3600.0, "dt": 20.0})
    print(f"GOR={r['GOR']:.2f}  distillate={r['distillate_total_m3_h']:.1f} m3/h  "
          f"steam={r['steam_flow_kg_s']:.2f} kg/s  recovery={r['recovery']:.3f}")
    print(f"T cascade (degC): {r['T_effect'].round(1)}")
    print(f"Transient T_top settled to {r['transient']['T_top']:.1f} C, "
          f"GOR(t_end)={r['transient']['GOR'][-1]:.2f}")
