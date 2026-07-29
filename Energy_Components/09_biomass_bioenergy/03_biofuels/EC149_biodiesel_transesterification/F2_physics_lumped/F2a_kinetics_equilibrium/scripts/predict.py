"""
EC149 -- Biodiesel Transesterification -- F2a Physics-Lumped Kinetics
Standardised predict() / get_info() interface (mirrors EC001 F2a).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import TransesterificationF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC149 F2a kinetics + reactor energy balance model."""

    component_id = "EC149"
    component_name = "Biodiesel Transesterification"
    fidelity = "F2a -- Physics-Lumped 3-Step Reversible Kinetics + Reactor Energy Balance ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = TransesterificationF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run dynamic batch transesterification simulation.

        inputs:
            TG0_mol_L       : float  initial triglyceride conc. (default 1.0)
            methanol_ratio  : float  MeOH:oil molar ratio       (default 6.0)
            T0_K            : float  initial reactor temperature (default 333.15)
            catalyst_factor : float  rate scaling (NaOH proxy)   (default 1.0)
            duration_min    : float  reaction time               (default 90.0)
            n_points        : int    output samples              (default 120)
            isothermal      : bool   disable energy balance       (default False)
        """
        TG0 = inputs.get("TG0_mol_L", 1.0)
        ratio = inputs.get("methanol_ratio", 6.0)
        T0 = inputs.get("T0_K", 333.15)
        cat = inputs.get("catalyst_factor", 1.0)
        dur = inputs.get("duration_min", 90.0)
        npts = inputs.get("n_points", 120)
        iso = inputs.get("isothermal", False)

        return self._model.simulate(
            TG0=TG0, methanol_ratio=ratio, T0=T0,
            catalyst_factor=cat, duration_min=dur, n_points=npts,
            isothermal=iso,
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "TG0_mol_L": {"unit": "mol/L", "range": [0.1, 2.0]},
                "methanol_ratio": {"unit": "mol_MeOH/mol_oil", "range": [3.0, 12.0]},
                "T0_K": {"unit": "K", "range": [303.15, 343.15]},
                "catalyst_factor": {"unit": "-", "range": [0.1, 5.0]},
                "duration_min": {"unit": "min", "range": [1.0, 240.0]},
                "n_points": {"unit": "-"},
                "isothermal": {"unit": "bool"},
            },
            "outputs": {
                "t": "min",
                "TG": "mol/L", "DG": "mol/L", "MG": "mol/L",
                "FAME": "mol/L", "glycerol": "mol/L", "methanol": "mol/L",
                "temperature": "K",
                "conversion": "- (TG conversion fraction)",
                "FAME_yield": "- (fraction of 3 FAME per TG max)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    info = m.get_info()
    print(f"{info['component_id']} {info['component_name']} -- {info['fidelity']}")
    r = m.predict({"methanol_ratio": 6.0, "T0_K": 333.15, "duration_min": 90.0})
    print(f"Final FAME yield: {r['FAME_yield_final']*100:.1f} %, "
          f"TG conversion: {r['conversion_final']*100:.1f} %, "
          f"T_final: {r['T_final']:.2f} K")
    print(f"Residual TG: {r['TG'][-1]:.4f} mol/L, "
          f"FAME: {r['FAME_final']:.4f} mol/L, glycerol: {r['glycerol'][-1]:.4f} mol/L")
