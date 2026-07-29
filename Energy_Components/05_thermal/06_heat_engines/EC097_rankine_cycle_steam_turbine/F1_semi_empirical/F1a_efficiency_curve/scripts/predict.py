"""
EC097 — Rankine Cycle (Steam Turbine) — F1a — Standardized Predict Interface

Usage:
    model = ComponentModel()
    result = model.predict({"PLR": 0.8})
"""

import json
import numpy as np
from pathlib import Path
from model import RankineCycleF1a


class ComponentModel:
    """Standardized interface for EC097 Rankine Cycle — F1a efficiency curve."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = RankineCycleF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "PLR": float or array (0-1), part-load ratio
                "T_steam": (optional) steam temperature in degC
                "T_condenser": (optional) condenser temperature in degC
            }
        Returns:
            {
                "efficiency": cycle efficiency,
                "power_output": electrical output in W,
                "heat_input": thermal input in W,
                "steam_flow": steam mass flow in kg/s,
                "carnot_efficiency": Carnot upper bound
            }
        """
        PLR = np.asarray(inputs["PLR"], dtype=float)
        T_s = inputs.get("T_steam", None)
        T_c = inputs.get("T_condenser", None)

        return {
            "efficiency": self._model.cycle_efficiency(PLR, T_s, T_c),
            "power_output": self._model.power_output(PLR),
            "heat_input": self._model.heat_input(PLR, T_s, T_c),
            "steam_flow": self._model.steam_mass_flow(PLR, T_s, T_c),
            "carnot_efficiency": self._model.carnot_efficiency(
                T_s if T_s is not None else self._model.T_steam,
                T_c if T_c is not None else self._model.T_condenser,
            ),
        }

    def get_info(self) -> dict:
        return {
            "name": "Rankine Cycle (Steam Turbine)",
            "ec_id": "EC097",
            "fidelity": "F1a",
            "description": "Part-load efficiency curve: eta = eta_rated*(1 - a*(1-PLR)^2)",
            "inputs": {
                "PLR": {"unit": "dimensionless", "range": [0.0, 1.0]},
                "T_steam": {"unit": "degC", "range": [400, 620], "optional": True},
                "T_condenser": {"unit": "degC", "range": [20, 50], "optional": True},
            },
            "outputs": {
                "efficiency": {"unit": "dimensionless"},
                "power_output": {"unit": "W"},
                "heat_input": {"unit": "W"},
                "steam_flow": {"unit": "kg/s"},
                "carnot_efficiency": {"unit": "dimensionless"},
            },
            "source": "Cotton (1998), Evaluating and Improving Steam Turbine Performance",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    result = model.predict({"PLR": 0.8})
    print(f"\nAt PLR=0.8:")
    print(f"  Efficiency: {result['efficiency']:.4f}")
    print(f"  Power: {result['power_output']/1e6:.1f} MW")
    print(f"  Heat input: {result['heat_input']/1e6:.1f} MW")
