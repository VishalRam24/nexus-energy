"""
EC218 -- Thermionic Converter -- F2a Physics-Lumped Emitter-Thermal
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import ThermionicF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for thermionic converter F2a physics-lumped model."""

    component_id = "EC218"
    component_name = "Thermionic Converter"
    fidelity = "F2a -- Physics-Lumped with Emitter-Temperature ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = ThermionicF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Integrate the lumped emitter-temperature ODE under a given heat input.

        inputs:
            Q_external_w  : float -- heat power into emitter [W]   (default 60.0)
            T_emitter0_K  : float -- initial emitter temperature [K] (default 1800)
            T_collector_K : float -- collector sink temperature [K]  (default 900)
            V_load        : float or None -- clamped terminal voltage [V] (default None)
            dt            : float -- output step [s] (default 0.05)
            duration_s    : float -- duration [s]   (default 20.0)
        """
        u = self._raw["unit"]
        Q = inputs.get("Q_external_w", u["Q_external_default"]["value"])
        T_e0 = inputs.get("T_emitter0_K", u["T_emitter_default"]["value"])
        T_c = inputs.get("T_collector_K", u["T_collector_default"]["value"])
        V_load = inputs.get("V_load", None)
        dt = inputs.get("dt", 0.05)
        dur = inputs.get("duration_s", 20.0)

        return self._model.simulate(Q, T_e0, T_c, dt, dur, V_load=V_load)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "Q_external_w": {"unit": "W", "range": [0, 300]},
                "T_emitter0_K": {"unit": "K", "range": [1200, 2200]},
                "T_collector_K": {"unit": "K", "range": [400, 1200]},
                "V_load": {"unit": "V", "range": [0, 1.0]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "T_emitter": "K",
                "J_net_Am2": "A/m^2",
                "V_terminal_V": "V",
                "power_w": "W",
                "power_density_w_cm2": "W/cm^2",
                "heat_input_w": "W",
                "efficiency": "-",
                "carnot_efficiency": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"Q_external_w": 60.0, "duration_s": 20.0, "dt": 0.5})
    print(
        f"Final T_emitter: {r['T_emitter'][-1]:.1f} K, "
        f"V_term: {r['V_terminal_V'][-1]:.3f} V, "
        f"P: {r['power_w'][-1]:.3f} W, "
        f"eta: {r['efficiency'][-1]:.3f} (Carnot {r['carnot_efficiency'][-1]:.3f})"
    )
