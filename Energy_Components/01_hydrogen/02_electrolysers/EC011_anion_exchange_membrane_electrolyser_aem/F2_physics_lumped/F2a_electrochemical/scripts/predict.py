"""
EC011 -- Anion Exchange Membrane (AEM) Electrolyser -- F2a Electrochemical
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import AEM_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for AEM electrolyser F2a full electrochemical model."""

    component_id = "EC011"
    component_name = "Anion Exchange Membrane Electrolyser (AEM)"
    fidelity = "F2a -- Full Electrochemical Model with Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = AEM_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic AEM electrolyser simulation.

        inputs:
            current_density_A_cm2 : float (or callable for time-varying)
            T_cell_K   : float  (initial temperature, default 313.15)
            P_h2_bar   : float  (default 1.0)
            P_o2_bar   : float  (default 1.0)
            dt         : float  (default 1.0)
            duration_s : float  (default 300.0)
        """
        j = inputs.get("current_density_A_cm2", 1.0)
        T0 = inputs.get("T_cell_K", 313.15)
        P_h2 = inputs.get("P_h2_bar", 1.0)
        P_o2 = inputs.get("P_o2_bar", 1.0)
        dt = inputs.get("dt", 1.0)
        dur = inputs.get("duration_s", 300.0)

        return self._model.simulate(j, T0, P_h2, P_o2, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_density_A_cm2": {"unit": "A/cm2", "range": [0, 3.0]},
                "T_cell_K": {"unit": "K", "range": [300, 353.15]},
                "P_h2_bar": {"unit": "bar", "range": [1.0, 30.0]},
                "P_o2_bar": {"unit": "bar", "range": [1.0, 30.0]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "voltage": "V (per cell)",
                "stack_voltage": "V",
                "power_kW": "kW",
                "h2_rate_mol_s": "mol/s",
                "efficiency": "- (HHV)",
                "faradaic_eff": "-",
                "temperature": "K",
                "overpotentials": "dict of V arrays",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"current_density_A_cm2": 1.0, "duration_s": 60.0, "dt": 5.0})
    print(f"Final cell V: {r['voltage'][-1]:.4f} V, "
          f"stack V: {r['stack_voltage'][-1]:.3f} V, "
          f"P: {r['power_kW'][-1]:.3f} kW, "
          f"H2: {r['h2_rate_mol_s'][-1]*1000:.4f} mmol/s, "
          f"eff(HHV): {r['efficiency'][-1]*100:.1f}%, "
          f"T: {r['temperature'][-1]:.2f} K")
