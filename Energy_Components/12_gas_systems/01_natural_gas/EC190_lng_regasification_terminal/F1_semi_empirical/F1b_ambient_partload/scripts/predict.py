"""EC190 — LNG Regasification Terminal — F1b Ambient+Part-Load — Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import LNGRegasF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = LNGRegasF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict LNG regasification energy with ambient and part-load corrections.

        Parameters
        ----------
        inputs : dict
            sendout_rate_ton_per_h : float (ton/h)
            PLR                    : float (0.2-1.0)
            T_ambient_K            : float (K, default 283.15)
        """
        m = inputs.get("sendout_rate_ton_per_h", 100.0)
        plr = np.asarray(inputs.get("PLR", 1.0), dtype=float)
        T_amb = inputs.get("T_ambient_K", 283.15)
        return self._model.compute(m, plr, T_amb)

    def get_info(self) -> dict:
        return {
            "name": "LNG Regasification Terminal",
            "ec_id": "EC190",
            "fidelity": "F1b",
            "description": (
                "SEC-based LNG regasification with ambient temperature correction "
                "(warmer seawater reduces SEC) and part-load penalty."
            ),
            "inputs": {
                "sendout_rate_ton_per_h": {"unit": "ton/h", "range": [10, 5000]},
                "PLR": {"unit": "dimensionless", "range": [0.2, 1.0]},
                "T_ambient_K": {"unit": "K", "range": [263, 308], "default": 283.15},
            },
            "outputs": {
                "gross_sec_kwh_per_ton": {"unit": "kWh/ton"},
                "net_sec_kwh_per_ton": {"unit": "kWh/ton"},
                "net_power_kw": {"unit": "kW"},
                "cold_recovery_kw": {"unit": "kW"},
                "gas_sendout_kg_per_s": {"unit": "kg/s"},
            },
            "source": "Mokhatab et al. (2014); Shah et al. (2013); DNV GL (2018)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"sendout_rate_ton_per_h": 100.0, "PLR": 1.0, "T_ambient_K": 283.15})
    print("Design point (PLR=1.0, T_amb=10°C):")
    for k, v in r.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")
    r2 = model.predict({"sendout_rate_ton_per_h": 100.0, "PLR": 0.5, "T_ambient_K": 273.15})
    print("\nPart-load (PLR=0.5, T_amb=0°C):")
    for k, v in r2.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")
