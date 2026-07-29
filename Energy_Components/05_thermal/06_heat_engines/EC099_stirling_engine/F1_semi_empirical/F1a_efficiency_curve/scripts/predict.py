"""
EC099 — Stirling Engine — F1a — Standardized Predict Interface
"""

import json
import numpy as np
from pathlib import Path
from model import StirlingEngineF1a


class ComponentModel:
    """Standardized interface for EC099 Stirling Engine — F1a efficiency curve."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = StirlingEngineF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "PLR": float or array (0-1), part-load ratio,
                "T_h": (optional) hot-side temperature in degC,
                "T_c": (optional) cold-side temperature in degC,
            }
        Returns:
            efficiency, power_output, heat_input, heat_rejected,
            carnot_efficiency
        """
        PLR = np.asarray(inputs["PLR"], dtype=float)
        T_h = inputs.get("T_h", None)
        T_c = inputs.get("T_c", None)

        return {
            "efficiency":        self._model.cycle_efficiency(PLR, T_h, T_c),
            "power_output":      self._model.power_output(PLR),
            "heat_input":        self._model.heat_input(PLR, T_h, T_c),
            "heat_rejected":     self._model.heat_rejected(PLR, T_h, T_c),
            "carnot_efficiency": self._model.carnot_efficiency(
                T_h if T_h is not None else self._model.T_h,
                T_c if T_c is not None else self._model.T_c,
            ),
        }

    def get_info(self) -> dict:
        return {
            "name": "Stirling Engine",
            "ec_id": "EC099",
            "fidelity": "F1a",
            "description": (
                "Stirling efficiency = f_carnot * eta_carnot * "
                "(1 - a*(1-PLR)^2), capped by Carnot."
            ),
            "inputs": {
                "PLR": {"unit": "dimensionless", "range": [0.0, 1.0]},
                "T_h": {"unit": "degC", "range": [400, 800], "optional": True},
                "T_c": {"unit": "degC", "range": [5, 80],    "optional": True},
            },
            "outputs": {
                "efficiency":        {"unit": "dimensionless"},
                "power_output":      {"unit": "W"},
                "heat_input":        {"unit": "W"},
                "heat_rejected":     {"unit": "W"},
                "carnot_efficiency": {"unit": "dimensionless"},
            },
            "source": (
                "Kongtragool & Wongwises (2003) RSER 7, 131-154; "
                "Cinar et al. (2005) Appl. Thermal Eng. 25, 1845-1854."
            ),
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    result = model.predict({"PLR": 0.8})
    print(f"\nAt PLR=0.8:")
    print(f"  Efficiency: {float(result['efficiency']):.4f}")
    print(f"  Power: {float(result['power_output'])/1e3:.2f} kW")
    print(f"  Heat input: {float(result['heat_input'])/1e3:.2f} kW")
    print(f"  Carnot: {float(result['carnot_efficiency']):.4f}")
