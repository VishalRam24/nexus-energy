"""EC216 — TEG — F1b Temperature-Dependent — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import TEGF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = TEGF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict TEG performance with temperature-dependent material properties.

        Parameters
        ----------
        inputs : dict
            T_hot_K  : float or array (K, 323-573)
            T_cold_K : float or array (K, 273-323)

        Returns
        -------
        dict with efficiency, power_density_w_cm2, zt_average, voltage_V
        """
        T_h = inputs.get("T_hot_K", 473.15)
        T_c = inputs.get("T_cold_K", 303.15)

        return self._model.compute(T_h, T_c)

    def get_info(self) -> dict:
        return {
            "name": "Thermoelectric Generator (TEG)",
            "ec_id": "EC216",
            "fidelity": "F1b",
            "description": (
                "Bi2Te3 TEG with temperature-dependent properties: "
                "alpha(T), k(T), sigma(T). ZT(T) computed locally and averaged "
                "across temperature gradient. Efficiency from standard ZT formula."
            ),
            "inputs": {
                "T_hot_K": {"unit": "K", "range": [323, 573], "default": 473.15},
                "T_cold_K": {"unit": "K", "range": [273, 323], "default": 303.15},
            },
            "outputs": {
                "efficiency": {"unit": "dimensionless"},
                "power_density_w_cm2": {"unit": "W/cm2"},
                "zt_average": {"unit": "dimensionless"},
                "voltage_V": {"unit": "V"},
            },
            "source": "Rowe (2006); Snyder & Toberer (2008)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_hot_K": 473.15, "T_cold_K": 303.15})
    print("Design point (T_hot=200C, T_cold=30C):")
    for k, v in r.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.6f}")
