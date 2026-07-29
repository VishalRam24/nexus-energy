"""
EC084 -- Aquifer Thermal Energy Storage (ATES) -- F1a Fully Mixed -- Standardized Predict Interface

Usage:
    model = ComponentModel()
    result = model.predict({"T_storage": 20.0, "m_dot": 5.0})
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import ATESModel

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    """Standardized interface for EC084 ATES -- F1a Fully Mixed model."""

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = _PARAMS_PATH
        with open(params_path) as f:
            raw = json.load(f)
        # Support both flat and nested formats
        p = raw.get("unit", raw)
        params = {}
        for k, v in p.items():
            params[k] = v["value"] if isinstance(v, dict) and "value" in v else v
        # Also check top-level keys
        for k in ["V_aquifer", "rho", "Cp", "eta_recovery", "T_ground", "T_max", "T_min"]:
            if k in raw:
                params[k] = raw[k]
        self.params = params
        self._model = ATESModel(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "T_storage": float -- mean aquifer temperature [degC],
                "m_dot": float -- mass flow rate [kg/s] (positive=extraction, default 0),
                "T_in": float -- inlet temperature [degC] (optional)
            }
        Returns: {
            "E_stored_J": J,
            "E_stored_kWh": kWh,
            "Q_thermal_W": W,
            "delta_T": K,
            "SOC": dimensionless,
            "T_storage": degC
        }
        """
        T_storage = float(inputs.get("T_storage", self._model.T_ground))
        m_dot = float(inputs.get("m_dot", 0.0))
        T_in = inputs.get("T_in", None)
        if T_in is not None:
            T_in = float(T_in)
        return self._model.evaluate(T_storage, m_dot=m_dot, T_in=T_in)

    def get_info(self) -> dict:
        return {
            "name": "Aquifer Thermal Energy Storage (ATES)",
            "ec_id": "EC084",
            "fidelity": "F1a",
            "description": "Fully-mixed aquifer: E=V*rho*Cp*dT*eta_recovery, V=50000m^3, eta=0.7",
            "inputs": {
                "T_storage": {"unit": "degC", "range": [5.0, 90.0]},
                "m_dot": {"unit": "kg/s", "range": [0.0, 100.0], "note": "positive=extraction"},
                "T_in": {"unit": "degC", "note": "inlet temperature (optional)"},
            },
            "outputs": {
                "E_stored_J": {"unit": "J"},
                "E_stored_kWh": {"unit": "kWh"},
                "Q_thermal_W": {"unit": "W"},
                "delta_T": {"unit": "K"},
                "SOC": {"unit": "dimensionless"},
                "T_storage": {"unit": "degC"},
            },
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    result = model.predict({"T_storage": 20.0, "m_dot": 5.0})
    print("\nAt T_storage=20degC, m_dot=5 kg/s:")
    for k, v in result.items():
        print(f"  {k}: {v}")
