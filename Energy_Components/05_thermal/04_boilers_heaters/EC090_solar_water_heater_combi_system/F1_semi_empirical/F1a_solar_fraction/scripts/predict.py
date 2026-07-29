"""
EC090 -- Solar Water Heater Combi System -- F1a Solar Fraction -- Standardized Predict Interface

Usage:
    model = ComponentModel()
    result = model.predict({"G_W_m2": 800.0})
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import SolarCombiModel

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    """Standardized interface for EC090 Solar Combi -- F1a solar fraction model."""

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = _PARAMS_PATH
        with open(params_path) as f:
            raw = json.load(f)
        self.params = raw.get("default_parameters", raw)
        self._model = SolarCombiModel(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "G_W_m2": float -- solar irradiance [W/m^2],
                "Q_demand_W": float -- heat demand [W] (optional, defaults to 10 kW)
            }
        Returns: {
            "Q_solar_W": W,
            "Q_aux_input_W": W (fuel energy to auxiliary boiler),
            "Q_aux_delivered_W": W (heat from auxiliary boiler),
            "f_solar": dimensionless (0-1),
            "Q_demand_W": W,
            "eta_system": dimensionless
        }
        """
        G = float(inputs.get("G_W_m2", 0.0))
        Q_demand = inputs.get("Q_demand_W", None)
        if Q_demand is not None:
            Q_demand = float(Q_demand)
        return self._model.evaluate(G, Q_demand_W=Q_demand)

    def get_info(self) -> dict:
        return {
            "name": "Solar Water Heater Combi System",
            "ec_id": "EC090",
            "fidelity": "F1a",
            "description": "Solar fraction model: Q_solar=eta_coll*G*A, Q_aux=(Q_demand-Q_solar)/eta_boiler",
            "inputs": {
                "G_W_m2": {"unit": "W/m^2", "range": [0.0, 1200.0]},
                "Q_demand_W": {"unit": "W", "note": "heat demand (default 10000 W)"},
            },
            "outputs": {
                "Q_solar_W": {"unit": "W"},
                "Q_aux_input_W": {"unit": "W"},
                "Q_aux_delivered_W": {"unit": "W"},
                "f_solar": {"unit": "dimensionless", "range": [0.0, 1.0]},
                "Q_demand_W": {"unit": "W"},
                "eta_system": {"unit": "dimensionless"},
            },
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    result = model.predict({"G_W_m2": 800.0})
    print("\nAt G=800 W/m^2:")
    for k, v in result.items():
        print(f"  {k}: {v}")
