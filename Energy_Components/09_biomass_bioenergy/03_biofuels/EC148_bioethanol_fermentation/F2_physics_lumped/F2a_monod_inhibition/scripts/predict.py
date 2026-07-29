"""
EC148 -- Bioethanol Fermentation -- F2a Monod + Luong Inhibition
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import BioethanolFermentationF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC148 F2a physics-lumped fermentation model."""

    component_id = "EC148"
    component_name = "Bioethanol Fermentation"
    fidelity = "F2a -- Monod Growth with Ethanol Product Inhibition + Lumped Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = BioethanolFermentationF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run batch fermentation simulation.

        inputs:
            S0_g_L      : float  initial glucose [g/L]        (default param S0)
            X0_g_L      : float  initial biomass [g/L]        (default param X0)
            P0_g_L      : float  initial ethanol [g/L]        (default 0)
            T0_K        : float  initial temperature [K]      (default T_opt)
            dt_h        : float  output step [h]              (default 0.25)
            duration_h  : float  fermentation time [h]        (default 48.0)
        """
        S0 = inputs.get("S0_g_L", None)
        X0 = inputs.get("X0_g_L", None)
        P0 = inputs.get("P0_g_L", None)
        T0 = inputs.get("T0_K", None)
        dt = inputs.get("dt_h", 0.25)
        dur = inputs.get("duration_h", 48.0)

        return self._model.simulate(S0, X0, P0, T0, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "S0_g_L": {"unit": "g/L", "range": [0, 350]},
                "X0_g_L": {"unit": "g/L", "range": [0.01, 20]},
                "P0_g_L": {"unit": "g/L", "range": [0, 120]},
                "T0_K": {"unit": "K", "range": [288.15, 318.15]},
                "dt_h": {"unit": "h"},
                "duration_h": {"unit": "h"},
            },
            "outputs": {
                "t": "h",
                "glucose": "g/L",
                "biomass": "g/L",
                "ethanol": "g/L",
                "temperature": "K",
                "mu": "1/h",
                "ethanol_yield_g_g": "g/g (<= 0.511 theoretical)",
                "productivity_g_L_h": "g/(L.h)",
                "ferment_efficiency": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"duration_h": 48.0, "dt_h": 0.5})
    print(
        f"Final ethanol: {r['ethanol_final_g_L']:.2f} g/L | "
        f"yield: {r['ethanol_yield_g_g']:.3f} g/g | "
        f"productivity: {r['productivity_g_L_h']:.3f} g/(L.h) | "
        f"T_final: {r['T_final_K']:.2f} K"
    )
