"""
EC102 -- Kalina Cycle -- F2a Physics-Lumped Thermodynamic Cycle
Standardised predict() / get_info() interface (mirrors EC001 F2a template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import KalinaCycleF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the Kalina Cycle F2a physics-lumped model."""

    component_id = "EC102"
    component_name = "Kalina Cycle (KCS, NH3-H2O)"
    fidelity = "F2a -- Physics-Lumped Thermodynamic Cycle with Transient Drum ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = KalinaCycleF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Solve the Kalina cycle. Steady-state by default; if `transient=True`,
        integrate the lumped drum-temperature ODE.

        inputs:
            T_source_c : float  heat-source temperature [degC]
            T_sink_c   : float  heat-sink temperature [degC]
            x_NH3      : float  basic-solution NH3 mass fraction
            Q_in_kw    : float  heat input duty [kW] (optional -> rated power)
            transient  : bool   if True, run the ODE
            duration_s : float  transient horizon [s] (default 600)
            Q_source_kw: float  heat-source duty for transient [kW]
            T0_K       : float  initial drum temperature [K] (transient)
        """
        T_src = inputs.get("T_source_c", None)
        T_snk = inputs.get("T_sink_c", None)
        x = inputs.get("x_NH3", None)
        Q_in = inputs.get("Q_in_kw", None)

        ss = self._model.solve_cycle(T_source_c=T_src, T_sink_c=T_snk,
                                     w_basic=x, Q_in_kw=Q_in)

        if inputs.get("transient", False):
            dur = inputs.get("duration_s", 600.0)
            T0 = inputs.get("T0_K", None)
            q_src_kw = inputs.get("Q_source_kw", ss["Q_in_kW"])
            tr = self._model.simulate_transient(
                q_source_func=q_src_kw * 1e3,
                T0_K=T0, duration_s=dur, w_basic=x,
            )
            ss["transient"] = tr
        return ss

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_source_c": {"unit": "degC", "range": [80, 220]},
                "T_sink_c": {"unit": "degC", "range": [5, 45]},
                "x_NH3": {"unit": "-", "range": [0.30, 0.95]},
                "Q_in_kw": {"unit": "kW", "range": [10, 5000]},
                "transient": {"unit": "bool"},
                "duration_s": {"unit": "s"},
                "Q_source_kw": {"unit": "kW"},
                "T0_K": {"unit": "K"},
            },
            "outputs": {
                "P_net_kW": "kW",
                "Q_in_kW": "kW",
                "Q_out_kW": "kW",
                "eta_thermal": "-",
                "eta_carnot": "-",
                "vapor_fraction": "-",
                "w_NH3_vapor": "-",
                "w_NH3_liquid": "-",
                "glide_hot_K": "K",
                "glide_cold_K": "K",
                "transient": "dict (t, T_drum_K, P_net_kW, eta_thermal)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"T_source_c": 150.0, "T_sink_c": 25.0})
    print(f"\nSteady state @ 150C source / 25C sink:")
    print(f"  P_net      = {r['P_net_kW']:.2f} kW")
    print(f"  eta_thermal= {r['eta_thermal']:.4f}  (Carnot {r['eta_carnot']:.4f})")
    print(f"  glide hot  = {r['glide_hot_K']:.2f} K, glide cold = {r['glide_cold_K']:.2f} K")
    print(f"  vapor frac = {r['vapor_fraction']:.3f}, "
          f"w_NH3 vapor={r['w_NH3_vapor']:.3f} liquid={r['w_NH3_liquid']:.3f}")
    tr = m.predict({"T_source_c": 150.0, "transient": True, "duration_s": 300.0})["transient"]
    print(f"  transient: T_drum {tr['T_drum_K'][0]:.1f} -> {tr['T_drum_K'][-1]:.1f} K "
          f"(ode success={tr['success']})")
