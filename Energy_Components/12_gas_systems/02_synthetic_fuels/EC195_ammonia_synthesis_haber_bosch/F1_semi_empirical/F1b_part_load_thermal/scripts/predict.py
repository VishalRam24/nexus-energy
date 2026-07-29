"""EC195 — Ammonia Synthesis (Haber-Bosch) — F1b Part-Load Thermal — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import AmmoniaF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AmmoniaF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict Haber-Bosch performance at part-load.

        Parameters
        ----------
        inputs : dict
            n2_flow_mol_s : float (mol/s, default 1.0)
            h2_n2_ratio   : float (mol/mol, stoich=3, default 3.0)
            PLR           : float or array (0.3-1.0)
            pressure_bar  : float (default 200)
            temperature_c : float (default 450)

        Returns
        -------
        dict with nh3_production_mol_s, single_pass_conversion, recycle_ratio,
             energy_kwh_per_ton, purge_fraction
        """
        n2 = inputs.get("n2_flow_mol_s", 1.0)
        ratio = inputs.get("h2_n2_ratio", 3.0)
        plr = np.asarray(inputs.get("PLR", 1.0), dtype=float)
        P = inputs.get("pressure_bar", 200.0)
        T = inputs.get("temperature_c", 450.0)

        return self._model.compute(n2, ratio, plr, P, T)

    def get_info(self) -> dict:
        return {
            "name": "Ammonia Synthesis (Haber-Bosch)",
            "ec_id": "EC195",
            "fidelity": "F1b",
            "description": (
                "N2 + 3H2 -> 2NH3. Part-load: reduced pressure -> lower conversion. "
                "Recycle ratio = 1/X_sp - 1. Energy = compression + heating. Loop purge."
            ),
            "inputs": {
                "n2_flow_mol_s": {"unit": "mol/s", "default": 1.0},
                "h2_n2_ratio": {"unit": "mol/mol", "range": [2.5, 3.5], "default": 3.0},
                "PLR": {"unit": "dimensionless", "range": [0.3, 1.0], "default": 1.0},
                "pressure_bar": {"unit": "bar", "range": [100, 300], "default": 200},
                "temperature_c": {"unit": "degC", "range": [350, 550], "default": 450},
            },
            "outputs": {
                "nh3_production_mol_s": {"unit": "mol/s"},
                "single_pass_conversion": {"unit": "dimensionless"},
                "recycle_ratio": {"unit": "dimensionless"},
                "energy_kwh_per_ton": {"unit": "kWh/ton_NH3"},
                "purge_fraction": {"unit": "dimensionless"},
            },
            "source": "Appl (2011), Ullmann's; Patil et al. (2015)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"n2_flow_mol_s": 1.0, "h2_n2_ratio": 3.0, "PLR": 1.0})
    print("Design point (PLR=1.0, T=450C, P=200bar):")
    for k, v in r.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")
