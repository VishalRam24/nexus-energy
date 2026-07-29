"""EC205 — CO2 Electrolyzer — F1b Part-Load + Electrode Degradation — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import CO2ElectrolyzerF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CO2ElectrolyzerF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict CO2 electrolyzer performance at part-load with electrode degradation.

        Parameters
        ----------
        inputs : dict
            PLR              : float (0.25-1.0, default 1.0)
            operating_hours  : float (hours, default 0)
        """
        plr = np.asarray(inputs.get("PLR", 1.0), dtype=float)
        hours = inputs.get("operating_hours", 0.0)

        return self._model.compute(plr, hours)

    def get_info(self) -> dict:
        return {
            "name": "CO2 Electrolyzer (CO2RR to CO)",
            "ec_id": "EC205",
            "fidelity": "F1b",
            "description": (
                "CO2 reduction electrolyzer model with Faradaic efficiency degradation "
                "(1% per 1000h) and part-load voltage penalty. Outputs CO production, "
                "SEC, and cell voltage vs PLR and operating hours."
            ),
            "inputs": {
                "PLR": {"unit": "dimensionless", "range": [0.25, 1.0], "default": 1.0},
                "operating_hours": {"unit": "hours", "range": [0, 50000], "default": 0},
            },
            "outputs": {
                "co_production_rate_g_h": {"unit": "g/h"},
                "sec_kwh_t_co": {"unit": "kWh/tCO"},
                "cell_voltage_V": {"unit": "V"},
                "faradaic_efficiency": {"unit": "dimensionless"},
                "fe_relative_pct": {"unit": "%"},
            },
            "source": "Jouny et al. (2018); Higgins et al. (2019)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"PLR": 1.0, "operating_hours": 0})
    print("Design point (PLR=1.0, fresh electrode):")
    for k, v in r.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")
