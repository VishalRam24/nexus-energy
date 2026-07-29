"""EC143 — Biomass Gasifier — F1a Equilibrium — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import BiomassGasifierF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BiomassGasifierF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict syngas composition and performance metrics.

        Parameters
        ----------
        inputs : dict
            equivalence_ratio : float or array  (0.2–0.5)
            temperature       : float or array  (degC, 700–1000), optional

        Returns
        -------
        dict
            syngas_composition : dict of mole fractions (CO, H2, CO2, CH4, N2)
            lhv_syngas_mjnm3   : float or array  [MJ/Nm3]
            cold_gas_efficiency : float or array  [dimensionless]
        """
        ER = np.asarray(inputs["equivalence_ratio"], dtype=float)
        T  = inputs.get("temperature", None)
        if T is not None:
            T = np.asarray(T, dtype=float)

        comp = self._model.syngas_composition(ER, T)
        lhv  = self._model.lhv_syngas(ER, T)
        cge  = self._model.cold_gas_efficiency(ER, T)

        return {
            "syngas_composition": {k: float(v) if v.ndim == 0 else v.tolist()
                                   for k, v in comp.items()},
            "lhv_syngas_mjnm3": lhv,
            "cold_gas_efficiency": cge,
        }

    def get_info(self) -> dict:
        return {
            "name": "Biomass Gasifier",
            "ec_id": "EC143",
            "fidelity": "F1a",
            "description": (
                "Simplified equilibrium model: syngas composition linear in ER. "
                "CO = 0.22 - 0.15*(ER-0.25), H2 = 0.18 - 0.12*(ER-0.25), etc."
            ),
            "inputs": {
                "equivalence_ratio": {"unit": "dimensionless", "range": [0.2, 0.5]},
                "temperature": {"unit": "degC", "range": [700.0, 1000.0], "default": 800.0},
            },
            "outputs": {
                "syngas_composition": {"unit": "mole_fraction", "keys": ["CO", "H2", "CO2", "CH4", "N2"]},
                "lhv_syngas_mjnm3": {"unit": "MJ/Nm3"},
                "cold_gas_efficiency": {"unit": "dimensionless"},
            },
            "source": "Zainal et al. (2001), Energy Conversion and Management, 42(12), 1499-1515",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"equivalence_ratio": 0.25, "temperature": 800.0})
    comp = r["syngas_composition"]
    print(f"ER=0.25, T=800C:")
    print(f"  CO={comp['CO']:.3f}, H2={comp['H2']:.3f}, CO2={comp['CO2']:.3f}, "
          f"CH4={comp['CH4']:.3f}, N2={comp['N2']:.3f}")
    print(f"  Sum = {sum(comp.values()):.4f}")
    print(f"  LHV = {float(r['lhv_syngas_mjnm3']):.3f} MJ/Nm3")
    print(f"  CGE = {float(r['cold_gas_efficiency']):.3f}")
