"""EC102 — Kalina Cycle — F1b Part-Load + Condenser T + NH3 Fraction — Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import KalinaCycleF1b


class ComponentModel:
    """Standardized interface for EC102 Kalina Cycle — F1b model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = KalinaCycleF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "T_heat_source" : float or array [degC, 80-220]
                "T_condenser"   : float or array [degC, 15-55]
                "PLR"           : float or array [0.3-1.0] (optional, default 1.0)
                "x_NH3"         : float or array [0.7-0.95] (optional, default design)
                "heat_input_kw" : float or array [kW] (optional)
            }
        Returns: efficiency, eta_carnot, power_output_kw, heat_rejection_kw,
                 f_composition, f_condenser
        """
        T_hot  = np.asarray(inputs.get("T_heat_source", self._model.T_hot_design), dtype=float)
        T_cond = np.asarray(inputs.get("T_condenser", self._model.T_cond_design), dtype=float)
        PLR    = np.asarray(inputs.get("PLR", 1.0), dtype=float)
        x_NH3  = inputs.get("x_NH3", None)
        Q_hot  = inputs.get("heat_input_kw", None)
        return self._model.predict_all(T_hot, T_cond, PLR, x_NH3, Q_hot)

    def get_info(self) -> dict:
        return {
            "name": "Kalina Cycle",
            "ec_id": "EC102",
            "fidelity": "F1b",
            "model": "Carnot × eta_int × f_composition × f_condenser × f_PLR",
            "description": (
                "eta = eta_Carnot * eta_int * f_x * f_T * f_PLR; "
                "f_x = 1 + k_x*(x_NH3 - x_design); "
                "f_T = 1 - k_T*(T_cond - T_design); "
                "Kalina advantage: 10-20% better eta than ORC for 100-200C sources"
            ),
            "inputs": {
                "T_heat_source": {"unit": "degC", "range": [80, 220]},
                "T_condenser":   {"unit": "degC", "range": [15, 55]},
                "PLR":           {"unit": "-",    "range": [0.3, 1.0], "default": 1.0},
                "x_NH3":         {"unit": "-",    "range": [0.70, 0.95], "default": "design"},
                "heat_input_kw": {"unit": "kW",   "range": [50, 5000], "default": "auto"},
            },
            "outputs": {
                "efficiency":        {"unit": "-"},
                "eta_carnot":        {"unit": "-"},
                "power_output_kw":   {"unit": "kW"},
                "heat_rejection_kw": {"unit": "kW"},
                "f_composition":     {"unit": "-"},
                "f_condenser":       {"unit": "-"},
            },
            "source": "Bombarda et al. (2010), Appl. Thermal Eng. 30(2); Lolos & Rogdakis (2009), Energy 34(4)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC102 F1b — Kalina Cycle — Design point:")
    r = model.predict({"T_heat_source": 150.0, "T_condenser": 32.0, "PLR": 1.0})
    for k, v in r.items():
        print(f"  {k}: {float(np.atleast_1d(v)[0]):.4f}")
