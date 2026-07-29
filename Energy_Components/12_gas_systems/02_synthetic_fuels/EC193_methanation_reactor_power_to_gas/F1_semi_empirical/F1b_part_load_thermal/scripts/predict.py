"""EC193 — Methanation Reactor — F1b Part-Load Thermal — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import MethanationF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MethanationF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict methanation performance at part-load with heat recovery.

        Parameters
        ----------
        inputs : dict
            co2_flow_mol_s  : float or array  (mol/s)
            h2_co2_ratio    : float or array  (mol/mol, stoich=4)
            PLR             : float or array  (0.3-1.0)
            T_reactor_degC  : float (default 300)
            pressure_bar    : float (default 10)

        Returns
        -------
        dict with ch4_production_mol_s, conversion, heat_recovery_kw,
             overall_efficiency, selectivity
        """
        co2 = np.asarray(inputs.get("co2_flow_mol_s", 1.0), dtype=float)
        ratio = np.asarray(inputs.get("h2_co2_ratio", 4.0), dtype=float)
        plr = np.asarray(inputs.get("PLR", 1.0), dtype=float)
        T = inputs.get("T_reactor_degC", 300.0)
        P = inputs.get("pressure_bar", 10.0)

        return self._model.compute(co2, ratio, plr, T, P)

    def get_info(self) -> dict:
        return {
            "name": "Methanation Reactor (Power-to-Gas)",
            "ec_id": "EC193",
            "fidelity": "F1b",
            "description": (
                "CO2 + 4H2 -> CH4 + 2H2O. Part-load conversion drop via PLR "
                "quadratic correction. Exothermic heat recovery Q = X*DH*n*f_recovery. "
                "Temperature-dependent selectivity."
            ),
            "inputs": {
                "co2_flow_mol_s": {"unit": "mol/s", "default": 1.0},
                "h2_co2_ratio": {"unit": "mol/mol", "range": [3.5, 5.0], "default": 4.0},
                "PLR": {"unit": "dimensionless", "range": [0.3, 1.0], "default": 1.0},
                "T_reactor_degC": {"unit": "degC", "range": [200, 500], "default": 300},
                "pressure_bar": {"unit": "bar", "range": [1, 30], "default": 10},
            },
            "outputs": {
                "ch4_production_mol_s": {"unit": "mol/s"},
                "conversion": {"unit": "dimensionless"},
                "heat_recovery_kw": {"unit": "kW"},
                "overall_efficiency": {"unit": "dimensionless"},
                "selectivity": {"unit": "dimensionless"},
            },
            "source": "Gao et al. (2012) RSC Adv; Gotz et al. (2016) Renew. Energy",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"co2_flow_mol_s": 1.0, "h2_co2_ratio": 4.0, "PLR": 1.0})
    print("Design point (PLR=1.0, T=300C, P=10bar):")
    for k, v in r.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")

    r2 = model.predict({"co2_flow_mol_s": 1.0, "h2_co2_ratio": 4.0, "PLR": 0.5})
    print("\nPart-load (PLR=0.5):")
    for k, v in r2.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")
