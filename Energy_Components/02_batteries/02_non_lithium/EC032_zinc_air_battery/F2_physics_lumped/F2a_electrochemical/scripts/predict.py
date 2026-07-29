"""
EC032 -- Zinc-Air Battery -- F2a Air-Cathode Electrochemical
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import ZincAirF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the Zn-air F2a air-cathode electrochemical model."""

    component_id = "EC032"
    component_name = "Zinc-Air Battery"
    fidelity = "F2a -- Air-Cathode Electrochemical Model with Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = ZincAirF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic Zn-air discharge simulation.

        inputs:
            current_density_A_cm2 : float (or callable for time-varying)
            T_cell_K  : float  (initial temperature, default 298.15)
            P_o2_atm  : float  (default 0.21 = ambient air)
            dt        : float  (default 1.0)
            duration_s: float  (default 600.0)
            soc0      : float  (default 1.0)
        """
        j = inputs.get("current_density_A_cm2", 0.05)
        T0 = inputs.get("T_cell_K", 298.15)
        P_o2 = inputs.get("P_o2_atm", 0.21)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 600.0)
        soc0 = inputs.get("soc0", 1.0)

        return self._model.simulate(j, T0, P_o2, dt, dur, soc0=soc0)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_density_A_cm2": {"unit": "A/cm2", "range": [0, 0.30]},
                "T_cell_K": {"unit": "K", "range": [253.15, 333.15]},
                "P_o2_atm": {"unit": "atm", "range": [0.05, 1.0]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
                "soc0": {"unit": "-", "range": [0, 1]},
            },
            "outputs": {
                "t": "s",
                "voltage": "V",
                "power_density": "W/cm2",
                "efficiency": "-",
                "temperature": "K",
                "soc": "-",
                "overpotentials": "dict of V arrays",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"current_density_A_cm2": 0.05, "duration_s": 600.0, "dt": 10.0})
    print(f"Plateau voltage: {r['voltage'][len(r['voltage'])//2]:.4f} V, "
          f"Final T: {r['temperature'][-1]:.2f} K, Final SOC: {r['soc'][-1]:.3f}")
