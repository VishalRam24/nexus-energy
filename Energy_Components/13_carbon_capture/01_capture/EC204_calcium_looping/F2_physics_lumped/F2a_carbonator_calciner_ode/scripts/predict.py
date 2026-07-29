"""
EC204 -- Calcium Looping -- F2a Carbonator/Calciner Coupled ODE
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import CalciumLoopingF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC204 calcium-looping F2a coupled-ODE model."""

    component_id = "EC204"
    component_name = "Calcium Looping (CaO/CaCO3)"
    fidelity = "F2a -- Coupled Carbonator/Calciner ODE with cyclic sorbent decay"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = CalciumLoopingF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the coupled carbonator/calciner dynamic simulation.

        inputs:
            cycle_number : float (or callable) -- sorbent age N [-], default 1
            T0_K         : float -- initial carbonator solids temp [K], default setpoint
            X0           : float -- initial sorbent conversion [-], default 0.0
            dt           : float -- output time step [s], default 1.0
            duration_s   : float -- total duration [s], default 300.0
        """
        N = inputs.get("cycle_number", 1)
        T0 = inputs.get("T0_K", None)
        X0 = inputs.get("X0", 0.0)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 300.0)

        return self._model.simulate(N, T0_K=T0, X0=X0, dt=dt, duration_s=dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "cycle_number": {"unit": "-", "range": [1, 1000]},
                "T0_K": {"unit": "K", "range": [820, 1000]},
                "X0": {"unit": "-", "range": [0.0, 0.8]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "conversion": "-",
                "capacity": "- (X_N Grasa-Abanades)",
                "capture_rate": "mol/s",
                "co2_out": "mol/s",
                "capture_efficiency": "-",
                "temperature": "K",
                "calciner_duty": "W",
                "carbon_balance_residual": "mol/s",
            },
            "source": self._raw["unit"].get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"cycle_number": 10, "duration_s": 300.0, "dt": 10.0})
    print(
        f"Fresh-loop X_N(N=10)={r['capacity'][-1]:.4f}, "
        f"final capture_eff={r['capture_efficiency'][-1]:.3f}, "
        f"final T={r['temperature'][-1]:.2f} K, "
        f"calciner duty={r['calciner_duty'][-1]/1e3:.1f} kW"
    )
