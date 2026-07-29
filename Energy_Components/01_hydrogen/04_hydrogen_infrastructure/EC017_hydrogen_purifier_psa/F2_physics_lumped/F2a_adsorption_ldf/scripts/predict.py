"""
EC017 -- Hydrogen Purifier (PSA) -- F2a Adsorption + LDF
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import HydrogenPSA_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC017 PSA F2a lumped adsorption model."""

    component_id = "EC017"
    component_name = "Hydrogen Purifier (Pressure Swing Adsorption, PSA)"
    fidelity = "F2a -- Lumped Adsorption Column (Langmuir + LDF) over PSA Cycle"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = HydrogenPSA_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a PSA cycle simulation (to cyclic steady state by default).

        inputs:
            feed_h2_fraction : float (default from params)
            feed_pressure_bar : float (adsorption / high pressure)
            purge_pressure_bar : float (blowdown / purge / low pressure)
            T_operating_K : float
            t_adsorption_s : float
            t_purge_s : float
            purge_to_feed_ratio : float
            dt : float (output/integration step, default 1.0 s)
            cyclic_steady_state : bool (default True) -- iterate to CSS
            n_cycles : int (max CSS iterations, default 20)
        """
        kw = dict(
            y_feed=inputs.get("feed_h2_fraction"),
            P_H=inputs.get("feed_pressure_bar"),
            P_L=inputs.get("purge_pressure_bar"),
            T=inputs.get("T_operating_K"),
            t_ads=inputs.get("t_adsorption_s"),
            t_purge=inputs.get("t_purge_s"),
            purge_ratio=inputs.get("purge_to_feed_ratio"),
            dt=inputs.get("dt", 1.0),
        )
        # drop None so model uses its defaults
        kw = {k: v for k, v in kw.items() if v is not None}

        if inputs.get("cyclic_steady_state", True):
            return self._model.cyclic_steady_state(
                n_cycles=inputs.get("n_cycles", 20), **kw
            )
        return self._model.simulate_cycle(q0=inputs.get("q0", 0.0), **kw)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "feed_h2_fraction": {"unit": "mol_fraction", "range": [0.30, 0.999]},
                "feed_pressure_bar": {"unit": "bar", "range": [5.0, 80.0]},
                "purge_pressure_bar": {"unit": "bar", "range": [1.0, 5.0]},
                "T_operating_K": {"unit": "K", "range": [273.15, 423.15]},
                "t_adsorption_s": {"unit": "s", "range": [10.0, 600.0]},
                "t_purge_s": {"unit": "s", "range": [10.0, 600.0]},
                "purge_to_feed_ratio": {"unit": "-", "range": [0.05, 0.40]},
                "dt": {"unit": "s"},
            },
            "outputs": {
                "t": "s",
                "loading": "mol/kg (solid-phase impurity q(t))",
                "loading_equilibrium": "mol/kg (Langmuir q* per step)",
                "purity": "- (H2 product mole fraction, (0,1])",
                "recovery": "- (H2 recovered / H2 fed, <1)",
                "productivity_mol_kg_cycle": "mol H2 / kg adsorbent / cycle",
                "specific_energy_kWh_per_kg_H2": "kWh/kg_H2",
                "impurity_balance_residual_mol": "mol (mass-conservation residual)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    info = m.get_info()
    print(f"{info['component_id']} {info['component_name']} [{info['fidelity']}]")
    r = m.predict({"feed_pressure_bar": 20.0, "feed_h2_fraction": 0.75})
    print(f"  CSS reached in {r['css_cycles']} cycles (q0={r['css_q0']:.4f} mol/kg)")
    print(f"  Purity       = {r['purity']*100:.4f} %  H2")
    print(f"  Recovery     = {r['recovery']*100:.2f} %")
    print(f"  Productivity = {r['productivity_mol_kg_cycle']:.4f} mol H2/kg/cycle")
    print(f"  Spec. energy = {r['specific_energy_kWh_per_kg_H2']:.3f} kWh/kg_H2")
    print(f"  Mass-balance residual = {r['impurity_balance_residual_mol']:.2e} mol")
