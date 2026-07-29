"""
EC049 -- Multi-Junction Concentrator PV (CPV) -- F2a Physics-Lumped
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import MultiJunctionCPV_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC049 multi-junction CPV F2a model."""

    component_id = "EC049"
    component_name = "Multi-Junction Concentrator PV (CPV)"
    fidelity = "F2a -- Series Multi-Junction Single-Diode + Current Matching + Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = MultiJunctionCPV_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a transient CPV simulation (thermal ODE + MPP at each step).

        inputs:
            DNI : float or callable(t)  [W/m2]  Direct Normal Irradiance
            T_amb_K : float  initial junction temperature [K] (default coolant temp)
            dt : float       [s] (default 1.0)
            duration_s : float [s] (default 120.0)
        """
        dni = inputs.get("DNI", 900.0)
        T0 = inputs.get("T_amb_K", None)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 120.0)
        return self._model.simulate(dni, T0, dt, dur)

    def predict_mpp(self, dni, T_K=None) -> dict:
        """Steady-state maximum power point at a single DNI / temperature."""
        if T_K is None:
            T_K = self._model.T_coolant
        return self._model.mpp(dni, T_K)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "DNI": {"unit": "W/m2", "range": [0, 1100]},
                "T_amb_K": {"unit": "K", "range": [243.15, 323.15]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "temperature": "K",
                "p_mp": "W",
                "v_mp": "V",
                "i_mp": "A",
                "v_oc": "V",
                "efficiency": "-",
                "concentration": "suns",
            },
            "subcells": self._model.names,
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    mpp = m.predict_mpp(900.0)
    print(f"\nSteady MPP @ DNI=900: P={mpp['p_mp']*1000:.2f} mW, "
          f"Voc={mpp['v_oc']:.3f} V, eta={mpp['efficiency']*100:.2f}%, "
          f"C={mpp['concentration']:.0f} suns, limiting={mpp['limiting_subcell']}")
    r = m.predict({"DNI": 900.0, "duration_s": 60.0, "dt": 2.0})
    print(f"Transient: T0={r['temperature'][0]:.2f} K -> "
          f"T_final={r['temperature'][-1]:.2f} K, "
          f"P_final={r['p_mp'][-1]*1000:.2f} mW")
