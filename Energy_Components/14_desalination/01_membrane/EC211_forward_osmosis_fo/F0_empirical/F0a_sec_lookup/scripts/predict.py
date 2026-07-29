"""Standardized prediction interface for the EC211 FO F0a SEC lookup."""
import json
import os
import numpy as np

try:
    from model import SECLookup
except ImportError:
    from .model import SECLookup

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARAMS = os.path.join(_HERE, "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC211"
    component_name = "Forward Osmosis (FO)"
    fidelity = "F0a - empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.params = json.load(f)
        u = self.params["unit"]
        self.lookup = SECLookup(
            load_bp=u["load_breakpoints"]["value"],
            mem_factor_bp=u["membrane_factor_breakpoints"]["value"],
            sec_membrane=u["sec_membrane_kWh_m3"]["value"],
            sec_regen=u["sec_regen_kWh_m3"]["value"],
            sec_total=u["sec_total_kWh_m3"]["value"],
            recovery=u["recovery"]["value"],
            rejection=u["rejection"]["value"],
            capacity_m3_h=u["capacity_m3_h"]["value"],
        )

    def predict(self, inputs: dict) -> dict:
        load = inputs.get("capacity_fraction", 1.0)
        include_regen = bool(inputs.get("include_regen", True))
        sec = self.lookup.sec(load, include_regen)
        perm = self.lookup.permeate_flow(load)
        return {
            "sec_kWh_m3": sec,
            "permeate_flow_m3h": perm,
            "power_kW": sec * perm,
            "include_regen": include_regen,
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "capacity_fraction": "dimensionless (0.1-1.0)",
                "include_regen": "boolean",
            },
            "outputs": {
                "sec_kWh_m3": "kWh/m3",
                "permeate_flow_m3h": "m3/hr",
                "power_kW": "kW",
            },
            "rated_sec_total_kWh_m3": self.lookup.sec_total,
            "source": self.params["unit"]["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print("sample @ full load + regen:", m.predict({"capacity_fraction": 1.0, "include_regen": True}))
    print("sample @ full load, membrane only:", m.predict({"capacity_fraction": 1.0, "include_regen": False}))
