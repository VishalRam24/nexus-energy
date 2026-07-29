"""EC100 — Brayton Cycle Gas Turbine — F1b Part-Load + Ambient — Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import BraytonGTF1b


class ComponentModel:
    """Standardized interface for EC100 Brayton GT — F1b model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BraytonGTF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "PLR"          : float or array [0.4-1.0]
                "T_ambient_k"  : float or array [K] (default ISO 288.15)
                "P_ambient_kpa": float or array [kPa] (optional, default 101.325)
            }
        Returns: efficiency, power_output_kw, heat_rate_kj_kwh,
                 exhaust_temp_k, f_amb_power
        """
        PLR   = np.asarray(inputs.get("PLR", 1.0), dtype=float)
        T_amb = np.asarray(inputs.get("T_ambient_k", self._model.T_iso_k), dtype=float)
        P_amb = np.asarray(inputs.get("P_ambient_kpa", self._model.P_iso_kpa), dtype=float)
        return self._model.predict_all(PLR, T_amb, P_amb)

    def get_info(self) -> dict:
        return {
            "name": "Brayton Cycle Gas Turbine (Simple Cycle)",
            "ec_id": "EC100",
            "fidelity": "F1b",
            "model": "Part-load + ISO 2314 ambient corrections",
            "description": (
                "eta = eta_rated * f_PLR(PLR) * sqrt(T_iso/T_amb); "
                "P = P_rated * PLR * (P_amb/P_iso) * sqrt(T_iso/T_amb); "
                "Exhaust T rises at part load"
            ),
            "inputs": {
                "PLR":           {"unit": "-",   "range": [0.4, 1.0]},
                "T_ambient_k":   {"unit": "K",   "range": [248, 323]},
                "P_ambient_kpa": {"unit": "kPa", "range": [80, 106], "default": 101.325},
            },
            "outputs": {
                "efficiency":       {"unit": "-"},
                "power_output_kw":  {"unit": "kW"},
                "heat_rate_kj_kwh": {"unit": "kJ/kWh"},
                "exhaust_temp_k":   {"unit": "K"},
                "f_amb_power":      {"unit": "-"},
            },
            "source": "Walsh & Fletcher (2004); ISO 2314:2009; GE Power F-class",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC100 F1b — Brayton GT — ISO conditions (PLR=1, T=288.15K):")
    r = model.predict({"PLR": 1.0, "T_ambient_k": 288.15})
    for k, v in r.items():
        print(f"  {k}: {float(np.atleast_1d(v)[0]):.4f}")
    print("\nHot day (T=308.15K, PLR=0.8):")
    r2 = model.predict({"PLR": 0.8, "T_ambient_k": 308.15})
    for k, v in r2.items():
        print(f"  {k}: {float(np.atleast_1d(v)[0]):.4f}")
