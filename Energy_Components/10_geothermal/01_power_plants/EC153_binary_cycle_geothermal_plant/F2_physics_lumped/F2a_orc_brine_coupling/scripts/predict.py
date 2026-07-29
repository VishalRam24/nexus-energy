"""
EC153 -- Binary Cycle Geothermal Plant -- F2a ORC-Brine Coupling
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import BinaryCycleGeothermal_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC153 Binary Cycle Geothermal F2a model."""

    component_id = "EC153"
    component_name = "Binary Cycle Geothermal Plant"
    fidelity = "F2a -- ORC-Brine Coupling with Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = BinaryCycleGeothermal_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic simulation.

        inputs:
            T_brine_in : float     Brine inlet temperature [K] (default 443.15)
            T_evap_init : float    Initial evaporator temperature [K] (default 393.15)
            dt : float             Output time step [s] (default 1.0)
            duration_s : float     Simulation duration [s] (default 300.0)
            brine_decline_years : float  Years of brine decline to simulate (default 0)
        """
        T_b = inputs.get("T_brine_in", self._raw["unit"]["T_brine_in"]["value"])
        T_ev0 = inputs.get("T_evap_init", self._raw["unit"]["T_evap"]["value"])
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 300.0)
        decline = inputs.get("brine_decline_years", 0.0)

        result = self._model.simulate(T_b, T_ev0, dt, dur,
                                       brine_decline_years=decline)
        return result

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_brine_in": {"unit": "K", "range": [373.15, 523.15]},
                "T_evap_init": {"unit": "K", "range": [313.15, 423.15]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
                "brine_decline_years": {"unit": "years", "range": [0, 30]},
            },
            "outputs": {
                "t": "s",
                "T_evap": "K",
                "T_brine_in": "K",
                "T_brine_out": "K",
                "W_net": "W",
                "W_turbine": "W",
                "W_parasitic": "W",
                "Q_in": "W",
                "eta_thermal": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"duration_s": 60.0, "dt": 5.0})
    print(f"Final W_net: {r['W_net'][-1]/1e6:.3f} MW, "
          f"eta_th: {r['eta_thermal'][-1]:.4f}, "
          f"T_brine_out: {r['T_brine_out'][-1]-273.15:.1f} C")
