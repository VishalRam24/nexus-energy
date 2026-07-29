"""
EC103 — Supercritical CO2 Brayton Cycle — F1a — Standardized Predict Interface
"""

import json
import numpy as np
from pathlib import Path
from model import SCO2BraytonF1a


class ComponentModel:
    """Standardized interface for EC103 sCO2 Brayton — F1a efficiency curve."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SCO2BraytonF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "PLR": float or array (0-1), part-load ratio,
                "T_in": (optional) turbine inlet temperature in degC,
                "T_reject": (optional) compressor inlet temperature in degC,
            }
        Returns:
            efficiency, power_output, heat_input, heat_rejected,
            carnot_efficiency
        """
        PLR = np.asarray(inputs["PLR"], dtype=float)
        T_in = inputs.get("T_in", None)
        T_rej = inputs.get("T_reject", None)

        return {
            "efficiency":        self._model.cycle_efficiency(PLR, T_in, T_rej),
            "power_output":      self._model.power_output(PLR),
            "heat_input":        self._model.heat_input(PLR, T_in, T_rej),
            "heat_rejected":     self._model.heat_rejected(PLR, T_in, T_rej),
            "carnot_efficiency": self._model.carnot_efficiency(
                T_in  if T_in  is not None else self._model.T_in,
                T_rej if T_rej is not None else self._model.T_reject,
            ),
        }

    def get_info(self) -> dict:
        return {
            "name": "Supercritical CO2 Brayton Cycle",
            "ec_id": "EC103",
            "fidelity": "F1a",
            "description": (
                "Recuperated sCO2 Brayton cycle: eta = eta_rated * "
                "(1 - a*(1-PLR)^2), capped by Carnot."
            ),
            "inputs": {
                "PLR":      {"unit": "dimensionless", "range": [0.0, 1.0]},
                "T_in":     {"unit": "degC", "range": [500, 800], "optional": True},
                "T_reject": {"unit": "degC", "range": [25, 60],   "optional": True},
            },
            "outputs": {
                "efficiency":        {"unit": "dimensionless"},
                "power_output":      {"unit": "W"},
                "heat_input":        {"unit": "W"},
                "heat_rejected":     {"unit": "W"},
                "carnot_efficiency": {"unit": "dimensionless"},
            },
            "source": (
                "Dostal et al. (2004) MIT-ANP-TR-100; "
                "Crespi et al. (2017) Appl. Energy 195, 152-183; "
                "Wright et al. (2010) SAND2010-0171."
            ),
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    result = model.predict({"PLR": 0.8})
    print(f"\nAt PLR=0.8:")
    print(f"  Efficiency: {float(result['efficiency']):.4f}")
    print(f"  Power: {float(result['power_output'])/1e6:.2f} MW")
    print(f"  Heat input: {float(result['heat_input'])/1e6:.2f} MW")
