"""
EC007 -- Reversible Fuel Cell (RFC) -- F2a Bidirectional Electrochemical
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import RFC_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for RFC F2a bidirectional electrochemical model."""

    component_id = "EC007"
    component_name = "Reversible Fuel Cell (RFC)"
    fidelity = "F2a -- Bidirectional Electrochemical Model with Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = RFC_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation. Sign of current density selects the mode:
        positive = discharge (fuel-cell), negative = charge (electrolysis).

        inputs:
            current_density_A_cm2 : float or callable(t)
            T_cell_K : float   (initial temperature, default 343.15)
            P_h2_atm : float   (default 1.0)
            P_o2_atm : float   (default 0.21)
            dt : float         (default 0.1)
            duration_s : float (default 60.0)
        """
        j = inputs.get("current_density_A_cm2", 0.5)
        T0 = inputs.get("T_cell_K", 343.15)
        P_h2 = inputs.get("P_h2_atm", 1.0)
        P_o2 = inputs.get("P_o2_atm", 0.21)
        dt = inputs.get("dt", 0.1)
        dur = inputs.get("duration_s", 60.0)

        return self._model.simulate(j, T0, P_h2, P_o2, dt, dur)

    def round_trip_efficiency(self, j_mag=0.5, T_cell_K=353.15, P_h2_atm=1.0, P_o2_atm=0.21):
        """Voltaic round-trip efficiency for a symmetric charge/discharge at |j|=j_mag."""
        return self._model.round_trip_efficiency(j_mag, T_cell_K, P_h2_atm, P_o2_atm)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_density_A_cm2": {"unit": "A/cm2", "range": [-3.0, 1.5],
                                          "note": "negative=charge(EL), positive=discharge(FC)"},
                "T_cell_K": {"unit": "K", "range": [300, 373.15]},
                "P_h2_atm": {"unit": "atm", "range": [0.5, 30.0]},
                "P_o2_atm": {"unit": "atm", "range": [0.1, 30.0]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "voltage": "V",
                "power_density": "W/cm2 (signed: + delivered / - consumed)",
                "efficiency": "-",
                "temperature": "K",
                "mode": "list[str] (FC/EL/OCV)",
                "overpotentials": "dict of V arrays",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    rfc = m.predict({"current_density_A_cm2": 0.6, "duration_s": 10.0, "dt": 1.0})
    print(f"[FC]  Final V: {rfc['voltage'][-1]:.4f} V, T: {rfc['temperature'][-1]:.2f} K, "
          f"mode={rfc['mode'][-1]}")
    rel = m.predict({"current_density_A_cm2": -0.6, "duration_s": 10.0, "dt": 1.0})
    print(f"[EL]  Final V: {rel['voltage'][-1]:.4f} V, T: {rel['temperature'][-1]:.2f} K, "
          f"mode={rel['mode'][-1]}")
    print(f"Round-trip efficiency @0.6 A/cm2: {m.round_trip_efficiency(0.6)*100:.1f} %")
