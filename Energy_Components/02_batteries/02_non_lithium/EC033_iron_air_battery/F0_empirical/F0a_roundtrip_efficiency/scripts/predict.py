"""Standardized inference API for Iron-Air Battery (EC033) F0a round-trip-efficiency lookup."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import RoundtripEfficiencyCurve

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC033"
    component_name = "Iron-Air Battery"
    fidelity = "F0a -- empirical lookup (round-trip efficiency curve)"
    version = "1.0.0"

    def __init__(self, params_path=_DATA):
        with open(params_path) as f:
            self.params = json.load(f)
        self.curve = RoundtripEfficiencyCurve(self.params)

    def predict(self, inputs: dict) -> dict:
        """inputs: {'c_rate': float|array} -> efficiency + loss + power_out estimate."""
        c_rate = inputs.get("c_rate", self.curve.c_rated)
        eta = self.curve.efficiency(c_rate)
        out = {
            "roundtrip_efficiency": float(eta) if np.isscalar(c_rate) or np.ndim(c_rate) == 0 else eta.tolist(),
            "loss_fraction": float(self.curve.loss_fraction(c_rate)) if np.ndim(c_rate) == 0 else self.curve.loss_fraction(c_rate).tolist(),
        }
        return out

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"c_rate": "C-rate (1/h), >=0"},
            "outputs": {"roundtrip_efficiency": "fraction (0-1)",
                        "loss_fraction": "fraction (0-1)"},
            "valid_ranges": {"c_rate": [0.0, 5.0]},
            "rated": {"eta_rated": self.curve.eta_rated, "c_rated": self.curve.c_rated},
            "source": self.params["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print("get_info:", json.dumps(m.get_info(), indent=2))
    for c in (0.0, m.curve.c_rated, 1.0, 5.0):
        print(f"  c_rate={c:>4} -> {m.predict({'c_rate': c})}")
