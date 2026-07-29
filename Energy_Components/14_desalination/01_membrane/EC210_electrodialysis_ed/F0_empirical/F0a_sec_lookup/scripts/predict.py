"""Standardized prediction interface for the EC210 ED F0a SEC-vs-load lookup."""
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
    component_id = "EC210"
    component_name = "Electrodialysis (ED)"
    fidelity = "F0a - empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.params = json.load(f)
        u = self.params["unit"]
        self.lookup = SECLookup(
            load_bp=u["load_breakpoints"]["value"],
            sec_bp=u["sec_breakpoints"]["value"],
            sec_rated=u["sec_rated"]["value"],
            recovery=u["recovery"]["value"],
            rejection=u["rejection"]["value"],
            capacity_m3_h=u["capacity_m3_h"]["value"],
        )

    def predict(self, inputs: dict) -> dict:
        load = inputs.get("capacity_fraction", 1.0)
        sec = self.lookup.sec(load)
        perm = self.lookup.permeate_flow(load)
        return {
            "sec_kWh_m3": sec,
            "permeate_flow_m3h": perm,
            "power_kW": sec * perm,
            "recovery": np.full_like(np.asarray(load, dtype=float), self.lookup.recovery),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"capacity_fraction": "dimensionless (0.1-1.0)"},
            "outputs": {
                "sec_kWh_m3": "kWh/m3",
                "permeate_flow_m3h": "m3/hr",
                "power_kW": "kW",
                "recovery": "dimensionless",
            },
            "rated_sec_kWh_m3": self.lookup.sec_rated,
            "source": self.params["unit"]["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print("sample @ load=1.0:", m.predict({"capacity_fraction": 1.0}))
    print("sample @ load=0.25:", m.predict({"capacity_fraction": 0.25}))
