"""EC210 — Electrodialysis (ED) — F1b Current Density + Donnan T — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import EDF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = EDF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict ED performance.

        Parameters
        ----------
        inputs : dict
            current_density     : float (A/m2)     default 100.0
            T_feed_degC         : float (degC)     default 25.0
            C_feed_mol_m3       : float (mol/m3)   default 100.0
            flow_rate_m3_h      : float (m3/h)     default 10.0
            operating_hours     : float (hours)    default 0.0
        """
        i     = np.asarray(inputs.get("current_density", 100.0), dtype=float)
        T     = inputs.get("T_feed_degC", 25.0)
        C     = inputs.get("C_feed_mol_m3", 100.0)
        Q     = inputs.get("flow_rate_m3_h", 10.0)
        hours = inputs.get("operating_hours", 0.0)

        return self._model.compute(i, T, C, Q, hours)

    def get_info(self) -> dict:
        return {
            "name": "Electrodialysis (ED)",
            "ec_id": "EC210",
            "fidelity": "F1b",
            "description": (
                "ED model with current density vs desalination rate coupling, "
                "Donnan equilibrium temperature dependence (co-ion exclusion), "
                "limiting current density effects, and membrane resistance aging (~10%/yr)."
            ),
            "inputs": {
                "current_density":  {"unit": "A/m2",   "range": [10, 280]},
                "T_feed_degC":      {"unit": "degC",   "range": [5, 45]},
                "C_feed_mol_m3":    {"unit": "mol/m3", "range": [5, 600]},
                "flow_rate_m3_h":   {"unit": "m3/h",   "range": [0.1, 1000]},
                "operating_hours":  {"unit": "hours",  "range": [0, 87600]},
            },
            "outputs": {
                "desalination_rate_mol_s":  {"unit": "mol/s"},
                "salinity_reduction_pct":   {"unit": "%"},
                "current_efficiency":       {"unit": "dimensionless"},
                "sec_kwh_m3":               {"unit": "kWh/m3"},
                "donnan_selectivity_factor": {"unit": "dimensionless"},
            },
            "source": "Strathmann (2010) Desalination; Campione et al. (2018) Desalination",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"current_density": 100.0, "T_feed_degC": 25.0,
                       "C_feed_mol_m3": 100.0, "flow_rate_m3_h": 10.0,
                       "operating_hours": 0})
    print("Design point (i=100 A/m2, T=25C, fresh membrane):")
    for k, v in r.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")
