"""
EC210 -- Electrodialysis (ED) -- F2a Ion-Transport Stack Model
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import ElectrodialysisF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for ED F2a ion-transport stack model."""

    component_id = "EC210"
    component_name = "Electrodialysis (ED)"
    fidelity = "F2a -- Ion-Transport Stack Model (Nernst-Planck + salt mass balance ODE)"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = ElectrodialysisF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a 1D plug-flow stack simulation.

        inputs:
            current_density_A_m2 : float (default 200.0)  -- applied current density
            feed_conc_mol_m3 : float (default 100.0)      -- diluate inlet conc.
            flow_velocity_cm_s : float (default 5.0)
            stack_length_cm : float (default 100.0)
            conc_feed_mol_m3 : float (default = feed)
            recovery_ratio : float (default 1.0)
            limiting_fraction : float (default 0.8)
        """
        i = inputs.get("current_density_A_m2", 200.0)
        c_feed = inputs.get("feed_conc_mol_m3", 100.0)
        v = inputs.get("flow_velocity_cm_s", 5.0)
        L = inputs.get("stack_length_cm", 100.0)
        cc = inputs.get("conc_feed_mol_m3", None)
        rr = inputs.get("recovery_ratio", 1.0)
        lf = inputs.get("limiting_fraction", 0.8)

        return self._model.simulate(
            current_density_A_m2=i,
            feed_conc_mol_m3=c_feed,
            flow_velocity_cm_s=v,
            stack_length_cm=L,
            conc_feed_mol_m3=cc,
            recovery_ratio=rr,
            limiting_fraction=lf,
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "current_density_A_m2": {"unit": "A/m2", "range": [10, 500]},
                "feed_conc_mol_m3": {"unit": "mol/m3", "range": [10, 600]},
                "flow_velocity_cm_s": {"unit": "cm/s", "range": [1, 20]},
                "stack_length_cm": {"unit": "cm", "range": [10, 200]},
                "conc_feed_mol_m3": {"unit": "mol/m3"},
                "recovery_ratio": {"unit": "-"},
                "limiting_fraction": {"unit": "-", "range": [0.1, 1.0]},
            },
            "outputs": {
                "x": "cm",
                "c_diluate": "mol/m3",
                "c_concentrate": "mol/m3",
                "current_density_local": "A/cm2",
                "limiting_current_density": "A/cm2",
                "stack_voltage": "V",
                "SEC_kWh_m3": "kWh/m3",
                "current_efficiency": "-",
                "salt_removed_fraction": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({
        "current_density_A_m2": 200.0,
        "feed_conc_mol_m3": 100.0,
        "flow_velocity_cm_s": 5.0,
        "stack_length_cm": 100.0,
    })
    print(
        f"Salt removed: {r['salt_removed_fraction']*100:.1f}% | "
        f"product={r['product_concentration_mol_m3']:.2f} mol/m3 | "
        f"SEC={r['SEC_kWh_m3']:.3f} kWh/m3 | "
        f"current_eff={r['current_efficiency']:.3f} | "
        f"U_stack(out)={r['stack_voltage'][-1]:.1f} V | "
        f"below i_lim={r['below_limiting_current']}"
    )
