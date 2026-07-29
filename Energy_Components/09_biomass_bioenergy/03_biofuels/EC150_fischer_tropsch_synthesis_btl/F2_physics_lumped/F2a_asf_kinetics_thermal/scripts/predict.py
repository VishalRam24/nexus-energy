"""
EC150 -- Fischer-Tropsch Synthesis (BTL) -- F2a ASF Kinetics + Thermal ODE
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import FischerTropschF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC150 FT F2a kinetics + exothermic thermal model."""

    component_id = "EC150"
    component_name = "Fischer-Tropsch Synthesis (BTL)"
    fidelity = "F2a -- ASF Kinetics + Lumped Exothermic Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = FischerTropschF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run FT reactor dynamic simulation (CO conversion + product slate + thermal).

        inputs:
            syngas_flow_mol_s : float (or callable for time-varying)  [mol/s]
            CO_fraction       : float   CO mole fraction in syngas (default 0.40)
            T0_K              : float   initial reactor temperature   (default 493.15)
            P_total_bar       : float   reactor pressure              (default 25.0)
            dt                : float   output step [s]               (default 10.0)
            duration_s        : float   total duration [s]            (default 3000.0)
        """
        Q = inputs.get("syngas_flow_mol_s", 100.0)
        CO_frac = inputs.get("CO_fraction", 0.40)
        T0 = inputs.get("T0_K", 493.15)
        P = inputs.get("P_total_bar", self._model.P_nom)
        dt = inputs.get("dt", 10.0)
        dur = inputs.get("duration_s", 3000.0)

        return self._model.simulate(Q, CO_frac, T0, P, dt, dur)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "syngas_flow_mol_s": {"unit": "mol/s", "range": [0, 1000]},
                "CO_fraction": {"unit": "-", "range": [0.2, 0.6]},
                "T0_K": {"unit": "K", "range": [460, 560]},
                "P_total_bar": {"unit": "bar", "range": [10, 40]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "temperature": "K",
                "CO_conversion": "-",
                "alpha": "-",
                "heat_generated_W": "W",
                "heat_removed_W": "W",
                "liquid_C5plus_kg_s": "kg/s",
                "energy_output_MW": "MW",
                "product_cuts": "dict of weight-fraction arrays",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"syngas_flow_mol_s": 100.0, "CO_fraction": 0.40,
                   "T0_K": 480.0, "duration_s": 2000.0, "dt": 50.0})
    print(f"Final T: {r['temperature'][-1]:.2f} K | "
          f"CO conversion: {r['CO_conversion'][-1]:.3f} | "
          f"alpha: {r['alpha'][-1]:.3f} | "
          f"C5+ liquid: {r['liquid_C5plus_kg_s'][-1]:.4f} kg/s | "
          f"energy: {r['energy_output_MW'][-1]:.3f} MW")
