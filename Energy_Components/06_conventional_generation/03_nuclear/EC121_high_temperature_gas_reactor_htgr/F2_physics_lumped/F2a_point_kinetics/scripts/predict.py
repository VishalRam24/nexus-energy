"""
EC121 -- High Temperature Gas Reactor (HTGR) -- F2a Point Kinetics + Lumped Thermal
Standardised predict() / get_info() interface (mirrors EC001 F2a).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import HTGR_F2a, PCM

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for HTGR F2a point-kinetics + lumped thermal model."""

    component_id = "EC121"
    component_name = "High Temperature Gas Reactor (HTGR)"
    fidelity = "F2a -- Point Reactor Kinetics + Lumped Fuel/Graphite/Helium Thermal"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = HTGR_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic reactor simulation.

        inputs:
            rho_ext_pcm : float (or callable(t)->pcm) external reactivity [pcm]
                          (default 0.0 = critical at rated power)
            duration_s  : float (default 1000.0)
            P0_fraction : float (default 1.0) initial power fraction
            n_eval      : int   (default 400) output samples
        """
        rho_pcm = inputs.get("rho_ext_pcm", 0.0)
        dur = inputs.get("duration_s", 1000.0)
        P0 = inputs.get("P0_fraction", 1.0)
        n_eval = inputs.get("n_eval", 400)

        if callable(rho_pcm):
            rho_ext = lambda t: rho_pcm(t) * PCM
        else:
            rho_ext = float(rho_pcm) * PCM

        return self._model.simulate(rho_ext, dur, P0, n_eval)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "rho_ext_pcm": {"unit": "pcm", "range": [-2000, 200]},
                "duration_s": {"unit": "s", "range": [1, 100000]},
                "P0_fraction": {"unit": "-", "range": [0.0, 1.2]},
                "n_eval": {"unit": "-"},
            },
            "outputs": {
                "t": "s",
                "power_fraction": "-",
                "P_thermal_MW": "MW_th",
                "P_electric_MW": "MW_e",
                "T_fuel_K": "K",
                "T_graphite_K": "K",
                "T_helium_outlet_C": "degC",
                "reactivity_total": "dk/k",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    # Insert -200 pcm of reactivity; reactor settles to a new lower power.
    r = m.predict({"rho_ext_pcm": -200.0, "duration_s": 2000.0})
    print(f"Final power fraction: {r['power_fraction'][-1]:.4f}")
    print(f"Final fuel T: {r['T_fuel_K'][-1]-273.15:.1f} C, "
          f"He outlet: {r['T_helium_outlet_C'][-1]:.1f} C")
