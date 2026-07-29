"""EC216 — Thermoelectric Generator (TEG) — F1a ZT Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import TEGF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = TEGF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            T_hot  : float or array — hot side temperature [degC]
            T_cold : float or array — cold side temperature [degC]
        returns:
            efficiency   : dimensionless
            power_w      : W
            heat_input_w : W
            voltage_v    : V
        """
        T_h = np.asarray(inputs["T_hot"], dtype=float)
        T_c = np.asarray(inputs["T_cold"], dtype=float)
        return self._model.compute(T_h, T_c)

    def get_info(self) -> dict:
        return {
            "name": "Thermoelectric Generator (TEG)",
            "ec_id": "EC216",
            "fidelity": "F1a",
            "description": "eta = eta_Carnot*(sqrt(1+ZT)-1)/(sqrt(1+ZT)+T_c/T_h), P = alpha^2*dT^2/(4R)",
            "inputs": {
                "T_hot":  {"unit": "degC", "range": [50.0, 300.0]},
                "T_cold": {"unit": "degC", "range": [0.0, 50.0]},
            },
            "outputs": {
                "efficiency":   {"unit": "-"},
                "power_w":      {"unit": "W"},
                "heat_input_w": {"unit": "W"},
                "voltage_v":    {"unit": "V"},
            },
            "source": "Rowe (2006) Thermoelectrics Handbook; Snyder & Toberer (2008) Nature Materials",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_hot": 200.0, "T_cold": 30.0})
    print(f"TEG at 200/30C: eta={float(r['efficiency'])*100:.1f}%, "
          f"P={float(r['power_w']):.2f}W, "
          f"Q_in={float(r['heat_input_w']):.1f}W, "
          f"V={float(r['voltage_v']):.2f}V")
