"""EC217 — TEC — F1b Temperature-Dependent Properties — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import TECF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = TECF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict TEC performance with temperature-dependent material properties.

        Parameters
        ----------
        inputs : dict
            T_cold_K : float or array (K, 200-320) — cold-side temperature
            T_hot_K  : float or array (K, 280-400) — hot-side temperature
            I_A      : float or array (A, 0.01-20)  — drive current

        Returns
        -------
        dict with Q_cold_W, Q_hot_W, W_input_W, COP, COP_max_theoretical,
                  T_min_achievable_K, ZT_avg, V_module_V
        """
        T_c = inputs.get("T_cold_K", 263.15)
        T_h = inputs.get("T_hot_K", 308.15)
        I = inputs.get("I_A", 3.0)
        return self._model.compute(T_c, T_h, I)

    def get_info(self) -> dict:
        return {
            "name": "Thermoelectric Cooler (TEC)",
            "ec_id": "EC217",
            "fidelity": "F1b",
            "description": (
                "Bi2Te3 TEC with temperature-dependent properties: alpha(T), k(T), sigma(T). "
                "Includes contact resistance, Thomson heat correction, Peltier/Joule balance. "
                "Cooling power Q_cold = alpha*I*Tc - 0.5*I^2*R - K*dT - 0.5*Q_Thomson."
            ),
            "inputs": {
                "T_cold_K": {"unit": "K", "range": [200, 320], "default": 263.15},
                "T_hot_K": {"unit": "K", "range": [280, 400], "default": 308.15},
                "I_A": {"unit": "A", "range": [0.01, 20], "default": 3.0},
            },
            "outputs": {
                "Q_cold_W": {"unit": "W", "note": "Cooling power (heat pumped from cold side)"},
                "Q_hot_W": {"unit": "W", "note": "Heat rejected at hot side"},
                "W_input_W": {"unit": "W", "note": "Electrical input power"},
                "COP": {"unit": "-", "note": "Coefficient of performance = Q_cold/W_input"},
                "COP_max_theoretical": {"unit": "-", "note": "Max COP from ZT formula"},
                "T_min_achievable_K": {"unit": "K", "note": "Ioffe minimum cold-side temperature"},
                "ZT_avg": {"unit": "-", "note": "Figure of merit at average T"},
                "V_module_V": {"unit": "V", "note": "Module terminal voltage"},
            },
            "source": "Rowe (2006); Goldsmid (1986); Ioffe (1957)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_cold_K": 263.15, "T_hot_K": 308.15, "I_A": 3.0})
    print("Design point (T_cold=-10C, T_hot=35C, I=3A):")
    for k, v in r.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.6f}")
