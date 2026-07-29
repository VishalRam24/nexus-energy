"""
EC143 -- Biomass Gasifier -- F2a Chemical Equilibrium
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import BiomassGasifier_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for Biomass Gasifier F2a equilibrium model."""

    component_id = "EC143"
    component_name = "Biomass Gasifier"
    fidelity = "F2a -- Chemical Equilibrium Model"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = BiomassGasifier_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Solve equilibrium gasification.

        inputs:
            biomass_C : float (carbon mass fraction, default from params)
            biomass_H : float
            biomass_O : float
            biomass_N : float
            moisture_content : float (default 0.15)
            equivalence_ratio : float (default 0.30)
            temperature_K : float (default 1073.15)
        """
        T = inputs.get("temperature_K", None)
        ER = inputs.get("equivalence_ratio", None)
        moist = inputs.get("moisture_content", None)
        C = inputs.get("biomass_C", None)
        H = inputs.get("biomass_H", None)
        O = inputs.get("biomass_O", None)
        N = inputs.get("biomass_N", None)

        result = self._model.solve_equilibrium(
            T=T, ER=ER, moisture=moist,
            C_mass=C, H_mass=H, O_mass=O, N_mass=N,
        )
        return result

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "biomass_C": {"unit": "mass fraction", "range": [0.35, 0.60]},
                "biomass_H": {"unit": "mass fraction", "range": [0.03, 0.08]},
                "biomass_O": {"unit": "mass fraction", "range": [0.30, 0.50]},
                "biomass_N": {"unit": "mass fraction", "range": [0.00, 0.05]},
                "moisture_content": {"unit": "-", "range": [0.0, 0.40]},
                "equivalence_ratio": {"unit": "-", "range": [0.15, 0.50]},
                "temperature_K": {"unit": "K", "range": [973.15, 1373.15]},
            },
            "outputs": {
                "composition_dry_mol_pct": "dict of gas species mol%",
                "composition_wet_mol_pct": "dict of gas species mol%",
                "LHV_syngas_MJ_Nm3": "MJ/Nm3",
                "HHV_biomass_MJ_kg": "MJ/kg",
                "gas_yield_Nm3_per_kg": "Nm3/kg",
                "cold_gas_efficiency": "-",
                "H2_CO_ratio": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"temperature_K": 1073.15, "equivalence_ratio": 0.30})
    print(f"\nSyngas composition (dry mol%):")
    for species, val in r["composition_dry_mol_pct"].items():
        print(f"  {species}: {val:.2f}%")
    print(f"LHV: {r['LHV_syngas_MJ_Nm3']:.3f} MJ/Nm3")
    print(f"Cold gas efficiency: {r['cold_gas_efficiency']:.3f}")
    print(f"H2/CO ratio: {r['H2_CO_ratio']:.3f}")
