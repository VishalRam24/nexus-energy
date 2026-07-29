"""Standardized prediction interface for the EC116 F0a power-map lookup."""
import json
import os
import numpy as np

try:
    from model import PowerLookup
except ImportError:
    from .model import PowerLookup

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARAMS = os.path.join(_HERE, "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC116"
    component_name = "Pressurized Water Reactor (PWR)"
    fidelity = "F0a - empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.params = json.load(f)
        u = self.params["unit"]
        self.lookup = PowerLookup(
            load_bp=u["load_breakpoints"]["value"],
            pelec_bp=u["pelec_breakpoints"]["value"],
            P_thermal_mw=u["P_thermal_mw"]["value"],
            eta_rated=u["eta_rated"]["value"],
            load_min=u["load_min"]["value"],
        )

    def predict(self, inputs: dict) -> dict:
        load = inputs.get("load_factor", 1.0)
        P_e = self.lookup.power_elec(load)
        eta = self.lookup.efficiency(load)
        P_th = self.lookup.P_thermal_mw * np.asarray(load, dtype=float)
        return {
            "power_output_mw": P_e,
            "efficiency": eta,
            "thermal_power_mw": P_th,
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"load_factor": "dimensionless (0.5-1.0)"},
            "outputs": {
                "power_output_mw": "MW_e",
                "efficiency": "dimensionless net thermal-to-electric",
                "thermal_power_mw": "MW_th",
            },
            "rated_power_mw": self.lookup.pelec_bp[-1],
            "source": self.params["unit"]["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print("sample @ load=1.0:", m.predict({"load_factor": 1.0}))
    print("sample @ load=0.75:", m.predict({"load_factor": 0.75}))
