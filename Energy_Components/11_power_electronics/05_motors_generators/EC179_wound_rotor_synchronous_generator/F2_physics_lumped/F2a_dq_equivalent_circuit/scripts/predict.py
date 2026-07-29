"""
EC179 -- Wound Rotor Synchronous Generator -- F2a dq-frame Physics-Lumped
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import WRSyncGenF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for WRSG F2a dq-frame machine model."""

    component_id = "EC179"
    component_name = "Wound Rotor Synchronous Generator"
    fidelity = "F2a -- dq-frame (Park) machine + field excitation + swing-equation rotor ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = WRSyncGenF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """Compute generator operating point and (optionally) rotor swing transient.

        inputs:
            P_pu      : float   active power [pu] (default 0.85)
            Q_pu      : float   reactive power [pu] (default 0.30; +over-excited)
            Vt_pu     : float   terminal voltage [pu] (default 1.0)
            simulate  : bool    if True, run swing-equation transient (default False)
            P_step    : float   step disturbance in Pm [pu] (default None)
            t_step    : float   time of step [s] (default None)
            avr       : bool    enable AVR field control (default False)
            Vref_pu   : float   AVR voltage setpoint [pu] (default 1.0)
            duration_s: float   transient length (default 5.0)
            dt        : float   output step (default 0.005)
        """
        P = inputs.get("P_pu", 0.85)
        Q = inputs.get("Q_pu", 0.30)
        Vt = inputs.get("Vt_pu", 1.0)

        op = self._model.operating_point(P, Q, Vt)
        result = dict(op)

        # Stability / pull-out margin
        Pmax = self._model.pmax(op["Ef_pu"], Vt)
        result["Pmax_pu"] = float(Pmax)
        result["stability_margin_pu"] = float(Pmax - P)

        if inputs.get("simulate", False):
            sim = self._model.simulate_swing(
                Pm=op["P_mech_pu"],
                Ef=op["Ef_pu"],
                Vt=Vt,
                delta0=op["delta_rad"],
                duration_s=inputs.get("duration_s", 5.0),
                dt=inputs.get("dt", 0.005),
                avr=inputs.get("avr", False),
                Vref=inputs.get("Vref_pu", 1.0),
                P_step=inputs.get("P_step", None),
                t_step=inputs.get("t_step", None),
            )
            result["transient"] = sim

        return result

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "P_pu": {"unit": "pu", "range": [0.0, 1.0]},
                "Q_pu": {"unit": "pu", "range": [-0.6, 0.8]},
                "Vt_pu": {"unit": "pu", "range": [0.9, 1.1]},
                "simulate": {"unit": "bool"},
                "P_step": {"unit": "pu"},
                "t_step": {"unit": "s"},
                "avr": {"unit": "bool"},
                "Vref_pu": {"unit": "pu", "range": [0.9, 1.1]},
                "duration_s": {"unit": "s"},
                "dt": {"unit": "s"},
            },
            "outputs": {
                "Ef_pu": "pu (internal EMF)",
                "delta_deg": "deg (power angle)",
                "If_A": "A (field current)",
                "Q_pu": "pu (reactive power)",
                "over_excited": "bool",
                "efficiency": "-",
                "Pmax_pu": "pu (pull-out limit)",
                "stable": "bool (delta<90deg)",
                "transient": "dict of t/delta/omega/Pe arrays (if simulate)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"P_pu": 0.85, "Q_pu": 0.30, "simulate": True,
                   "P_step": 0.1, "t_step": 0.5, "duration_s": 3.0})
    print(f"Ef={r['Ef_pu']:.3f} pu, delta={r['delta_deg']:.2f} deg, "
          f"If={r['If_A']:.1f} A, over-excited={r['over_excited']}, "
          f"eta={r['efficiency']:.4f}, Pmax={r['Pmax_pu']:.3f} pu")
    tr = r["transient"]
    print(f"Transient: final delta={tr['delta_deg'][-1]:.2f} deg, "
          f"final omega={tr['omega_pu'][-1]:.5f} pu, stable={tr['stable']}")
