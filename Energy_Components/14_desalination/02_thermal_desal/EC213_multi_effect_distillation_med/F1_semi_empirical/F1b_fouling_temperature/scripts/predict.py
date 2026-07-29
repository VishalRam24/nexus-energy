"""EC213 — MED Distillation — F1b GOR + TBT + Scaling — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import MEDF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MEDF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict MED distillation performance.

        Parameters
        ----------
        inputs : dict
            TBT_degC                : float (degC)      default 70.0
            plr                     : float (0.3-1.0)   default 1.0
            steam_temperature_degC  : float (degC)      default None
        """
        TBT   = np.asarray(inputs.get("TBT_degC", 70.0), dtype=float)
        plr   = np.asarray(inputs.get("plr", 1.0), dtype=float)
        T_stm = inputs.get("steam_temperature_degC", None)

        return self._model.compute(TBT, plr, T_stm)

    def get_info(self) -> dict:
        return {
            "name": "Multi-Effect Distillation (MED)",
            "ec_id": "EC213",
            "fidelity": "F1b",
            "description": (
                "MED model with GOR vs TBT and N_effects, BPE correction, "
                "CaCO3 scaling risk above 70 degC, and part-load pumping SEC."
            ),
            "inputs": {
                "TBT_degC":               {"unit": "degC",         "range": [55, 75]},
                "plr":                    {"unit": "dimensionless", "range": [0.3, 1.0]},
                "steam_temperature_degC": {"unit": "degC",         "range": [65, 90]},
            },
            "outputs": {
                "gor":                {"unit": "kg/kg"},
                "thermal_sec_kwh_m3": {"unit": "kWh_th/m3"},
                "pump_sec_kwh_m3":    {"unit": "kWh_e/m3"},
                "total_sec_kwh_m3":   {"unit": "kWh/m3"},
                "scaling_risk_index": {"unit": "dimensionless [0-1]"},
                "bpr_total_degC":     {"unit": "degC"},
            },
            "source": "El-Dessouky et al. (2000); Ettouney (2006)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"TBT_degC": 70.0, "plr": 1.0})
    print("Design point (TBT=70C, PLR=1.0):")
    for k, v in r.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")
