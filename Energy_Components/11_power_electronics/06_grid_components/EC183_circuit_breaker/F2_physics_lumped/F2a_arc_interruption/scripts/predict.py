"""
EC183 -- Circuit Breaker -- F2a Cassie-Mayr Arc Interruption
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import CircuitBreakerArc_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC183 F2a arc interruption model."""

    component_id = "EC183"
    component_name = "Circuit Breaker"
    fidelity = "F2a -- Cassie-Mayr Arc Interruption (TRV / current-zero ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = CircuitBreakerArc_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Simulate a fault-interruption event.

        inputs:
            I_fault_kA  : float  prospective fault current [kA] (default = breaking cap.)
            phi         : float  source phase at t=0 [rad] (default 0.0)
            dt_us       : float  output step [us] (default 0.01)
            duration_ms : float  window length [ms] (default 20.0)
            Vpk         : float  optional source emf peak override [V]
        """
        return self._model.simulate(
            I_fault_kA=inputs.get("I_fault_kA", None),
            phi=inputs.get("phi", None),
            dt_us=inputs.get("dt_us", 0.01),
            duration_ms=inputs.get("duration_ms", 20.0),
            Vpk=inputs.get("Vpk", None),
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "I_fault_kA": {"unit": "kA", "range": [0.1, 50.0]},
                "phi": {"unit": "rad", "range": [0.0, 6.2832]},
                "dt_us": {"unit": "us", "range": [0.001, 1.0]},
                "duration_ms": {"unit": "ms", "range": [0.1, 100.0]},
                "Vpk": {"unit": "V"},
            },
            "outputs": {
                "t": "s",
                "current": "A",
                "arc_voltage": "V",
                "conductance": "S",
                "arc_energy_total_J": "J",
                "trv_peak_V": "V",
                "n_current_zeros": "-",
                "interruption_success": "bool",
                "interruption_time_s": "s",
                "within_capacity": "bool",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"I_fault_kA": 20.0, "duration_ms": 15.0})
    print(
        f"I_fault={r['I_fault_kA']:.1f} kA | zeros={r['n_current_zeros']} | "
        f"arc_E={r['arc_energy_total_J']/1e3:.2f} kJ | "
        f"TRV_peak={r['trv_peak_V']/1e3:.1f} kV | "
        f"success={r['interruption_success']}"
    )
