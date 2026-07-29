"""
EC203 -- Membrane-Based CO2 Separation -- F2a Solution-Diffusion Cross-Flow
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import MembraneF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC203 F2a membrane CO2 separation module."""

    component_id = "EC203"
    component_name = "Membrane-Based CO2 Separation"
    fidelity = "F2a -- Solution-Diffusion Cross-Flow Module (1D area ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = MembraneF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a single-stage cross-flow membrane separation.

        inputs (all optional -- fall back to parameters.json):
            area_m2          : float  total membrane area
            feed_flow_mol_s  : float  total molar feed
            y_CO2_feed       : float  feed CO2 mole fraction
            n_eval           : int    number of area sample points

        If inputs contains 'two_stage': {area1_m2, area2_m2} it runs the
        two-stage cascade instead.

        Returns dict with recovery, purity, stage_cut, retentate fraction and
        full area-resolved profiles.
        """
        if "two_stage" in inputs:
            ts = inputs["two_stage"]
            return self._model.two_stage(
                area1_m2=ts.get("area1_m2", self._model.area),
                area2_m2=ts.get("area2_m2", self._model.area),
                feed_flow_mol_s=inputs.get("feed_flow_mol_s"),
                y_CO2_feed=inputs.get("y_CO2_feed"),
            )

        result = self._model.simulate(
            area_m2=inputs.get("area_m2"),
            feed_flow_mol_s=inputs.get("feed_flow_mol_s"),
            y_CO2_feed=inputs.get("y_CO2_feed"),
            n_eval=int(inputs.get("n_eval", 200)),
        )
        return result

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "area_m2": {"unit": "m2", "range": [1.0, 100000.0]},
                "feed_flow_mol_s": {"unit": "mol/s", "range": [0.001, 1000.0]},
                "y_CO2_feed": {"unit": "mol_fraction", "range": [0.001, 0.99]},
                "n_eval": {"unit": "-"},
                "two_stage": {"unit": "dict{area1_m2, area2_m2}"},
            },
            "outputs": {
                "recovery": "-",
                "purity": "-",
                "stage_cut": "-",
                "retentate_CO2_fraction": "-",
                "permeate_flow_mol_s": "mol/s",
                "pressure_ratio": "-",
                "area": "m2 array",
                "permeate_purity": "- array vs area",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"area_m2": 50.0})
    print(f"\nSingle-stage @ A=50 m2:")
    print(f"  CO2 recovery   = {r['recovery']*100:.1f} %")
    print(f"  permeate purity= {r['purity']*100:.1f} % CO2")
    print(f"  stage cut      = {r['stage_cut']*100:.1f} %")
    print(f"  pressure ratio = {r['pressure_ratio']:.1f}")
    ts = m.predict({"two_stage": {"area1_m2": 100.0, "area2_m2": 30.0}})
    print(f"\nTwo-stage cascade:")
    print(f"  overall recovery = {ts['overall_recovery']*100:.1f} %")
    print(f"  final purity     = {ts['final_purity']*100:.1f} % CO2")
