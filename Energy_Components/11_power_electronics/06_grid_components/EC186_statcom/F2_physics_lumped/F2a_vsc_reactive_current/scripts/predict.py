"""
EC186 -- STATCOM -- F2a VSC Reactive-Current Control (Physics-Lumped)
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import STATCOM_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for STATCOM F2a VSC reactive-current model."""

    component_id = "EC186"
    component_name = "STATCOM (Static Synchronous Compensator)"
    fidelity = "F2a -- VSC Reactive-Current Control with DC-Link ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = STATCOM_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic STATCOM simulation.

        inputs:
            Q_ref_MVAR : float | callable(t)->MVAR  reactive command (+ capacitive)
            V_bus_pu   : float | callable(t)->pu    bus voltage (default 1.0)
            dt         : float  output step [s]      (default 1e-4)
            duration_s : float  sim time [s]         (default 0.1)
        """
        Q_in = inputs.get("Q_ref_MVAR", 0.0)
        V_bus = inputs.get("V_bus_pu", 1.0)
        dt = inputs.get("dt", 1e-4)
        dur = inputs.get("duration_s", 0.1)

        # convert MVAR command to VAR (support callable)
        if callable(Q_in):
            Q_ref = lambda t: Q_in(t) * 1e6
        else:
            Q_ref = float(Q_in) * 1e6

        return self._model.simulate(Q_ref, V_bus, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "Q_ref_MVAR": {"unit": "MVAR", "range": [-100, 100]},
                "V_bus_pu": {"unit": "pu", "range": [0.2, 1.2]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "i_d": "A (peak, real current)",
                "i_q": "A (peak, reactive current)",
                "I_mag": "A (peak line current)",
                "Vdc": "V (DC-link voltage)",
                "Q_out_MVAR": "MVAR delivered to bus (+ capacitive)",
                "P_loss_W": "W converter losses",
                "V_conv_V": "V converter AC voltage (line-line RMS)",
                "V_bus_pu": "pu bus voltage",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"Q_ref_MVAR": 80.0, "V_bus_pu": 1.0, "dt": 1e-4, "duration_s": 0.05})
    print(f"Final Q_out: {r['Q_out_MVAR'][-1]:.2f} MVAR  "
          f"(ref 80 MVAR), Vdc: {r['Vdc'][-1]/1e3:.2f} kV, "
          f"I_mag: {r['I_mag'][-1]:.1f} A")
