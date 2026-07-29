"""EC096 — Magnetic Refrigeration — F1b COP + Part-Load — Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import MagneticRefrigerationF1b


class ComponentModel:
    """Standardized interface for EC096 Magnetic Refrigeration — F1b model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MagneticRefrigerationF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "T_hot_degC"  : float or array [degC] — hot reservoir temperature
                "T_cold_degC" : float or array [degC] — cold reservoir temperature
                "PLR"         : float or array [0.3-1.0] (optional, default 1.0)
            }
        Returns: cop, cop_carnot, eta_vs_carnot, cooling_kw, electrical_kw,
                 heat_rejection_kw, delta_T_span_K
        """
        T_hot  = np.asarray(inputs.get("T_hot_degC", self._model.T_hot_design), dtype=float)
        T_cold = np.asarray(inputs.get("T_cold_degC", self._model.T_cold_design), dtype=float)
        PLR    = np.asarray(inputs.get("PLR", 1.0), dtype=float)
        return self._model.predict_all(T_hot, T_cold, PLR)

    def get_info(self) -> dict:
        return {
            "name": "Magnetic Refrigeration (AMR)",
            "ec_id": "EC096",
            "fidelity": "F1b",
            "model": "Carnot COP × eta_AMR × f_PLR(PLR) × f_T(T_hot)",
            "description": (
                "COP = COP_Carnot * eta_AMR * f_PLR * f_T; "
                "eta_AMR accounts for magnet, regen, cycle losses; "
                "f_PLR = p1+p2*PLR+p3*PLR^2; f_T = 1 - k_T*(T_hot - T_design)"
            ),
            "inputs": {
                "T_hot_degC":  {"unit": "degC", "range": [25, 55]},
                "T_cold_degC": {"unit": "degC", "range": [5, 25]},
                "PLR":         {"unit": "-",    "range": [0.3, 1.0]},
            },
            "outputs": {
                "cop":               {"unit": "-"},
                "cop_carnot":        {"unit": "-"},
                "eta_vs_carnot":     {"unit": "-"},
                "cooling_kw":        {"unit": "kW"},
                "electrical_kw":     {"unit": "kW"},
                "heat_rejection_kw": {"unit": "kW"},
                "delta_T_span_K":    {"unit": "K"},
            },
            "source": "Kitanovski et al. (2015); Yu et al. (2010), Int. J. Refrig. 33(6)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC096 F1b — Magnetic Refrigeration — Design point:")
    r = model.predict({"T_hot_degC": 35.0, "T_cold_degC": 15.0, "PLR": 1.0})
    for k, v in r.items():
        print(f"  {k}: {float(np.atleast_1d(v)[0]):.4f}")
