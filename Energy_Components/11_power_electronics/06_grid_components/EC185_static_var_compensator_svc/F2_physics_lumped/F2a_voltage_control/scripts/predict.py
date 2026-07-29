"""
EC185 -- Static VAR Compensator (SVC) -- F2a Physics-Lumped Voltage Control
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import SVC_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for SVC F2a physics-lumped voltage-control model."""

    component_id = "EC185"
    component_name = "Static VAR Compensator (SVC)"
    fidelity = "F2a -- Physics-Lumped TCR/TSC with V-Q Droop Voltage-Control ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = SVC_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a closed-loop bus-voltage-regulation transient.

        inputs:
            E_thev      : float  Thevenin source voltage [pu] (default 1.05 -- over-voltage)
            X_thev      : float  source short-circuit reactance [pu] (default 0.1)
            V_ref       : float  regulator setpoint [pu] (default param value)
            B0          : float  initial susceptance [pu] (default mid-range)
            dt          : float  output step [s] (default 0.002)
            duration_s  : float  total time [s] (default 0.5)
        """
        E_thev = inputs.get("E_thev", 1.05)
        X_thev = inputs.get("X_thev", 0.10)
        V_ref = inputs.get("V_ref", None)
        B0 = inputs.get("B0", None)
        dt = inputs.get("dt", 0.002)
        dur = inputs.get("duration_s", 0.5)

        return self._model.simulate(E_thev, X_thev, dt, dur,
                                    V_ref=V_ref, B0=B0)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "E_thev": {"unit": "pu", "range": [0.85, 1.15]},
                "X_thev": {"unit": "pu", "range": [0.02, 0.5]},
                "V_ref": {"unit": "pu", "range": [0.9, 1.1]},
                "B0": {"unit": "pu", "range": [self._model.B_svc_min, self._model.B_svc_max]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "V_bus": "pu",
                "B_act": "pu (cap. positive)",
                "alpha_deg": "deg (TCR firing angle)",
                "Q_MVAR": "MVAR (cap. positive)",
                "Q_pu": "pu",
                "P_loss_MW": "MW",
                "mode": "capacitive/inductive/floating",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"E_thev": 1.05, "X_thev": 0.10, "duration_s": 0.5, "dt": 0.002})
    print(f"Over-voltage (E=1.05) regulation: V_bus {r['V_bus'][0]:.4f} -> "
          f"{r['V_bus'][-1]:.4f} pu, Q_final={r['Q_MVAR'][-1]:.2f} MVAR "
          f"(alpha={r['alpha_deg'][-1]:.1f} deg, mode={r['mode']})")
