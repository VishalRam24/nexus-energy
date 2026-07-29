"""
EC023 -- LMO Battery (Lithium Manganese Oxide) -- F0a predict interface.

Standardized ComponentModel wrapping the F0a round-trip-efficiency lookup.
Run standalone:  python3 scripts/predict.py
"""

import json
import os

from model import LMOBatteryF0a


class ComponentModel:
    component_id = "EC023"
    component_name = "LMO Battery (Lithium Manganese Oxide)"
    fidelity = "F0a -- empirical lookup (round-trip efficiency curve)"
    version = "1.0.0"

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        with open(params_path) as f:
            self.params = json.load(f)
        self.model = LMOBatteryF0a(self.params)

    def predict(self, inputs: dict) -> dict:
        """inputs: {'c_rate': float, optional 'energy_in_wh': float} -> outputs dict."""
        c_rate = inputs.get("c_rate", 0.0)
        eta = float(self.model.round_trip_efficiency(c_rate))
        out = {
            "round_trip_efficiency": eta,
            "loss_fraction": 1.0 - eta,
        }
        if "energy_in_wh" in inputs:
            out["usable_energy_wh"] = float(self.model.usable_energy(c_rate, inputs["energy_in_wh"]))
        return out

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "c_rate": "C-rate (dimensionless, 0..3.0)",
                "energy_in_wh": "optional energy charged in, Wh",
            },
            "outputs": {
                "round_trip_efficiency": "0-1",
                "loss_fraction": "0-1",
                "usable_energy_wh": "Wh (if energy_in_wh given)",
            },
            "valid_ranges": {"c_rate": [0.0, 3.0]},
            "source": 'LMO SOC-only F1a default parameters (Q_nom=2.5 Ah, R_int=30 mOhm, V 3.0-4.2 V)',
        }


if __name__ == "__main__":
    cm = ComponentModel()
    print(cm.get_info()["component_name"], "|", cm.fidelity)
    print("0.5C :", cm.predict({"c_rate": 0.5, "energy_in_wh": 100.0}))
    print("rated:", cm.predict({"c_rate": 0.0}))
