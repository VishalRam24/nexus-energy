"""
EC118 -- Small Modular Reactor (SMR) -- F2a Point Kinetics + Lumped Thermal
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import SMRPointKineticsThermalF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the SMR F2a point-kinetics + thermal model."""

    component_id = "EC118"
    component_name = "Small Modular Reactor (SMR)"
    fidelity = "F2a -- Point Kinetics (6-group) + Lumped Thermal, Doppler/Moderator Feedback, Load-Following"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = SMRPointKineticsThermalF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """Run a dynamic SMR transient.

        Two modes:
          * Reactivity transient (default): supply 'rho_ext' (scalar/callable),
            or 'rho_step' + 't_insert' for a step insertion.
          * Load-following: set inputs['mode'] = 'load_follow' and supply
            'power_schedule' (callable(t)->fraction) or a constant
            'power_demand' fraction.

        Common inputs:
            dt          : float, output step [s]      (default 0.5)
            duration_s  : float, total duration [s]   (default 100.0)
            n0          : float, initial power frac   (default 1.0)
        """
        dt = inputs.get("dt", 0.5)
        dur = inputs.get("duration_s", 100.0)
        n0 = inputs.get("n0", 1.0)
        x0 = self._model.initial_conditions(n0)

        mode = inputs.get("mode", "reactivity")

        if mode == "load_follow":
            sched = inputs.get("power_schedule")
            if sched is None:
                pd = inputs.get("power_demand", 1.0)
                sched = (lambda t: pd) if not callable(pd) else pd
            return self._model.load_follow(
                sched, dt=dt, duration_s=dur, x0=x0,
                kp=inputs.get("kp", 0.02), ki=inputs.get("ki", 2e-4),
            )

        if "rho_step" in inputs:
            return self._model.step_reactivity_insertion(
                inputs["rho_step"], dt=dt, duration_s=dur,
                t_insert=inputs.get("t_insert", 1.0), x0=x0,
            )

        rho_ext = inputs.get("rho_ext", 0.0)
        return self._model.simulate(rho_ext, dt, dur, x0=x0)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "rho_ext": {"unit": "dk/k", "range": [-0.01, 0.005]},
                "rho_step": {"unit": "dk/k", "range": [-0.01, 0.005]},
                "power_demand": {"unit": "-", "range": [0.2, 1.0]},
                "mode": {"options": ["reactivity", "load_follow"]},
                "n0": {"unit": "-", "range": [0.2, 1.0]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "n": "normalized neutron power (1.0 = rated)",
                "power_fraction": "-",
                "T_f": "K (lumped fuel temperature)",
                "T_m": "K (lumped coolant/moderator temperature)",
                "P_thermal_W": "W",
                "P_elec_W": "W",
                "rho": "dk/k (total reactivity incl. feedback)",
                "flow_fraction": "- (natural-circulation coolant flow / rated)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    # Step reactivity insertion of +100 pcm; feedback should arrest the rise.
    r = m.predict({"rho_step": 0.001, "duration_s": 60.0, "dt": 0.5})
    print(f"Step +100 pcm -> peak power frac {r['n'].max():.3f}, "
          f"settled {r['n'][-1]:.3f}, T_f {r['T_f'][-1]:.1f} K, "
          f"T_m {r['T_m'][-1]:.1f} K")
    # Load-following 100% -> 60% -> 100%.
    def sched(t):
        if t < 2000: return 1.0
        if t < 5000: return 0.6
        return 1.0
    rl = m.predict({"mode": "load_follow", "power_schedule": sched,
                    "duration_s": 8000.0, "dt": 5.0})
    print(f"Load-follow tracking error (max) "
          f"{abs(rl['power_fraction']-rl['power_demand']).max():.3f}")
