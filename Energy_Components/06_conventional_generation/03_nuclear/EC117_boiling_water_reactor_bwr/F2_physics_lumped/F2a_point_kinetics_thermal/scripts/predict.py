"""
EC117 -- Boiling Water Reactor (BWR) -- F2a Point Kinetics + Thermal-Hydraulics
Standardised predict() / get_info() interface (mirrors EC001 F2a template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import BWR_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for BWR F2a point-kinetics + thermal-hydraulic model."""

    component_id = "EC117"
    component_name = "Boiling Water Reactor (BWR)"
    fidelity = "F2a -- Point Reactor Kinetics + Lumped Thermal-Hydraulics with Reactivity Feedback"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = BWR_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a dynamic reactor transient.

        inputs:
            reactivity_dollars : float  external reactivity step in dollars (default 0.0).
                                 Mutually exclusive with reactivity_dkk.
            reactivity_dkk     : float  external reactivity in absolute dk/k (optional).
            n0                 : float  initial power fraction (default 1.0).
            duration_s         : float  transient length [s] (default 100.0).
            dt                 : float  output step [s] (default 0.05).
        """
        n0 = inputs.get("n0", 1.0)
        dt = inputs.get("dt", 0.05)
        dur = inputs.get("duration_s", 100.0)

        if "reactivity_dkk" in inputs:
            rho_ext = float(inputs["reactivity_dkk"])
        else:
            rho_ext = self._model.dollars_to_reactivity(inputs.get("reactivity_dollars", 0.0))

        return self._model.simulate(rho_ext, duration_s=dur, dt=dt, n0=n0)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "reactivity_dollars": {"unit": "$", "range": [-5.0, 0.9]},
                "reactivity_dkk": {"unit": "dk/k", "note": "alternative to dollars"},
                "n0": {"unit": "-", "range": [0.0, 1.2]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "n": "power fraction (-)",
                "power_mw": "MW_th",
                "T_fuel": "K",
                "T_coolant": "K",
                "void_fraction": "-",
                "reactivity": "dk/k",
                "reactivity_dollars": "$",
                "precursors": "(6, N) normalised precursor concentrations",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    # +0.5$ reactivity insertion from full power -- feedback should arrest the rise.
    r = m.predict({"reactivity_dollars": 0.5, "duration_s": 60.0, "dt": 0.05})
    print(f"Initial power: {r['power_mw'][0]:.1f} MW_th")
    print(f"Peak power:    {r['power_mw'].max():.1f} MW_th")
    print(f"Final power:   {r['power_mw'][-1]:.1f} MW_th")
    print(f"Final T_fuel:  {r['T_fuel'][-1]:.1f} K, void: {r['void_fraction'][-1]:.3f}")
    print(f"Final reactivity: {r['reactivity_dollars'][-1]:.3f} $")
