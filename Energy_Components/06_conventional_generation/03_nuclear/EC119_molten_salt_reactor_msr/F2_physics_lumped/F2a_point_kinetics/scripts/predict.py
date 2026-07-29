"""
EC119 -- Molten Salt Reactor (MSR) -- F2a Point Reactor Kinetics
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import MSR_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for MSR F2a flowing-fuel point-kinetics model."""

    component_id = "EC119"
    component_name = "Molten Salt Reactor (MSR)"
    fidelity = "F2a -- Flowing-Fuel Point Reactor Kinetics with Lumped Thermal"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = MSR_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a transient simulation.

        inputs:
            rho_ext_pcm   : float or callable(t)  external reactivity [pcm] (default 0)
            flow_fraction : float  fuel-salt flow / nominal (default 1.0)
            dt            : float  output step [s] (default 0.5)
            duration_s    : float  total time [s] (default 200.0)
            T_core0_K     : float  optional initial core temp [K]
            T_loop0_K     : float  optional initial loop temp [K]
            n0            : float  initial normalised neutron pop. (default 1.0)
        """
        return self._model.simulate(
            rho_ext_pcm=inputs.get("rho_ext_pcm", 0.0),
            flow_fraction=inputs.get("flow_fraction", 1.0),
            dt=inputs.get("dt", 0.5),
            duration_s=inputs.get("duration_s", 200.0),
            T_core0=inputs.get("T_core0_K", None),
            T_loop0=inputs.get("T_loop0_K", None),
            n0=inputs.get("n0", 1.0),
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "rho_ext_pcm": {"unit": "pcm", "range": [-2000, 300]},
                "flow_fraction": {"unit": "-", "range": [0.0, 1.5]},
                "dt": {"unit": "s", "range": [0.01, 5.0]},
                "duration_s": {"unit": "s", "range": [1.0, 5000.0]},
                "T_core0_K": {"unit": "K", "range": [800, 1200]},
                "T_loop0_K": {"unit": "K", "range": [800, 1200]},
                "n0": {"unit": "-", "range": [0.0, 2.0]},
            },
            "outputs": {
                "t": "s",
                "n": "- (normalised neutron population)",
                "power_w": "W_th",
                "power_fraction": "-",
                "T_core_K": "K",
                "T_loop_K": "K",
                "reactivity_pcm": "pcm",
                "beta_eff": "- (flowing-fuel effective delayed fraction)",
                "beta_static": "- (static delayed fraction)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    # Reactivity step of +50 pcm, watch power rise then settle via T-feedback.
    r = m.predict({"rho_ext_pcm": 50.0, "duration_s": 200.0, "dt": 1.0})
    print(f"\nbeta_static = {r['beta_static']:.5f},  "
          f"beta_eff(flow=1) = {r['beta_eff']:.5f}  "
          f"({100*(1-r['beta_eff']/r['beta_static']):.1f}% loss to flow)")
    print(f"Final power fraction: {r['power_fraction'][-1]:.3f}")
    print(f"Final core T: {r['T_core_K'][-1]:.2f} K, "
          f"loop T: {r['T_loop_K'][-1]:.2f} K")
