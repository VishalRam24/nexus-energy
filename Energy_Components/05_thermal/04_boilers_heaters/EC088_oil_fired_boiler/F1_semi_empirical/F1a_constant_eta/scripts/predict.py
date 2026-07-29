"""
EC088 -- Oil-Fired Boiler -- F1a Constant Efficiency -- Standardized Predict Interface

Usage:
    model = ComponentModel()
    result = model.predict({"m_fuel_kg_s": 0.01})
    # or
    result = model.predict({"Q_demand_W": 200000.0})
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import OilBoilerModel

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    """Standardized interface for EC088 Oil-Fired Boiler -- F1a constant-eta model."""

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = _PARAMS_PATH
        with open(params_path) as f:
            raw = json.load(f)
        self.params = raw.get("default_parameters", raw)
        self._model = OilBoilerModel(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: one of:
                {"m_fuel_kg_s": float}  -- fuel flow [kg/s]
                {"Q_demand_W": float}   -- heat demand [W]
        Returns: {
            "Q_out_W": W,
            "Q_out_kW": kW,
            "m_fuel_kg_s": kg/s,
            "Q_in_W": W,
            "losses_W": W,
            "load_fraction": dimensionless,
            "eta": dimensionless
        }
        """
        m_fuel = inputs.get("m_fuel_kg_s", None)
        Q_demand = inputs.get("Q_demand_W", None)
        if m_fuel is not None:
            m_fuel = float(m_fuel)
        if Q_demand is not None:
            Q_demand = float(Q_demand)
        return self._model.evaluate(m_fuel_kg_s=m_fuel, Q_demand_W=Q_demand)

    def get_info(self) -> dict:
        return {
            "name": "Oil-Fired Boiler",
            "ec_id": "EC088",
            "fidelity": "F1a",
            "description": "Constant-eta model: Q_out=eta*m_fuel*LHV, eta=0.87, LHV=42.6 MJ/kg, P_rated=500 kW",
            "inputs": {
                "m_fuel_kg_s": {"unit": "kg/s", "note": "fuel mass flow (or use Q_demand_W)"},
                "Q_demand_W": {"unit": "W", "note": "heat demand (alternative to m_fuel_kg_s)"},
            },
            "outputs": {
                "Q_out_W": {"unit": "W"},
                "Q_out_kW": {"unit": "kW"},
                "m_fuel_kg_s": {"unit": "kg/s"},
                "Q_in_W": {"unit": "W"},
                "losses_W": {"unit": "W"},
                "load_fraction": {"unit": "dimensionless"},
                "eta": {"unit": "dimensionless"},
            },
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    result = model.predict({"Q_demand_W": 250000.0})
    print("\nAt Q_demand=250 kW:")
    for k, v in result.items():
        print(f"  {k}: {v}")
