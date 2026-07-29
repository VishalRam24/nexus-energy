"""EC204 — Calcium Looping — F1b Part-Load + Sorbent Degradation — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import CalciumLoopingF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CalciumLoopingF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict calcium looping performance at part-load with cyclic sorbent degradation.

        Parameters
        ----------
        inputs : dict
            flue_gas_flow_mol_s : float (mol/s, default 100)
            co2_concentration   : float (mol/mol, default 0.12)
            capture_rate        : float (0.5-0.95, default 0.90)
            PLR                 : float (0.3-1.0, default 1.0)
            n_cycles            : float (cycles, default 0)
        """
        fg = inputs.get("flue_gas_flow_mol_s", 100.0)
        xCO2 = inputs.get("co2_concentration", 0.12)
        cr = inputs.get("capture_rate", 0.90)
        plr = np.asarray(inputs.get("PLR", 1.0), dtype=float)
        n_cycles = inputs.get("n_cycles", 0.0)

        return self._model.compute(fg, xCO2, cr, plr, n_cycles)

    def get_info(self) -> dict:
        return {
            "name": "Calcium Looping (CaL)",
            "ec_id": "EC204",
            "fidelity": "F1b",
            "description": (
                "CaL CO2 capture with Grasa-Abanades sorbent deactivation model and "
                "part-load calcination duty penalty. Activity: X(N)=X_inf+(X0-X_inf)*exp(-K*N)."
            ),
            "inputs": {
                "flue_gas_flow_mol_s": {"unit": "mol/s", "range": [10, 1000], "default": 100},
                "co2_concentration": {"unit": "mol/mol", "range": [0.04, 0.15], "default": 0.12},
                "capture_rate": {"unit": "dimensionless", "range": [0.5, 0.95], "default": 0.90},
                "PLR": {"unit": "dimensionless", "range": [0.3, 1.0], "default": 1.0},
                "n_cycles": {"unit": "cycles", "range": [0, 100000], "default": 0},
            },
            "outputs": {
                "co2_captured_kg_h": {"unit": "kg/h"},
                "calcination_duty_gj_ton": {"unit": "GJ/tCO2"},
                "electrical_kwh_ton": {"unit": "kWh/tCO2"},
                "sorbent_activity_pct": {"unit": "%"},
                "total_energy_penalty_pct": {"unit": "%"},
            },
            "source": "Grasa & Abanades (2006); Romano (2012)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"PLR": 1.0, "n_cycles": 0})
    print("Design point (PLR=1.0, fresh sorbent):")
    for k, v in r.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")
