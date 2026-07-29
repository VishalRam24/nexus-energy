"""Standardized prediction interface for the EC214 MVC F0a SEC-vs-CR curve."""
import json
import os
import numpy as np

try:
    from model import SECCurve
except ImportError:
    from .model import SECCurve

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARAMS = os.path.join(_HERE, "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC214"
    component_name = "Mechanical Vapor Compression (MVC)"
    fidelity = "F0a - empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.params = json.load(f)
        u = self.params["unit"]
        self.curve = SECCurve(
            cr_bp=u["cr_breakpoints"]["value"],
            sec_bp=u["sec_breakpoints"]["value"],
            cr_rated=u["cr_rated"]["value"],
            sec_rated=u["sec_rated_kWh_m3"]["value"],
            sec_min=u["sec_min_kWh_m3"]["value"],
            sec_max=u["sec_max_kWh_m3"]["value"],
            recovery=u["recovery"]["value"],
            capacity_m3_h=u["capacity_m3_h"]["value"],
        )

    def predict(self, inputs: dict) -> dict:
        cr = inputs.get("compression_ratio", self.curve.cr_rated)
        load = inputs.get("capacity_fraction", 1.0)
        sec = self.curve.sec(cr)
        dist = self.curve.distillate_flow(load)
        return {
            "sec_kWh_m3": sec,
            "distillate_flow_m3h": dist,
            "elec_power_kW": sec * dist,
            "recovery": np.full_like(np.asarray(sec, dtype=float), self.curve.recovery),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "compression_ratio": "dimensionless (1.05-1.5)",
                "capacity_fraction": "dimensionless (0.1-1.0)",
            },
            "outputs": {
                "sec_kWh_m3": "kWh/m3",
                "distillate_flow_m3h": "m3/hr",
                "elec_power_kW": "kW",
                "recovery": "dimensionless",
            },
            "rated_sec_kWh_m3": self.curve.sec_rated,
            "source": self.params["unit"]["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print("sample @ CR=1.2:", m.predict({"compression_ratio": 1.2}))
    print("sample @ CR=1.45:", m.predict({"compression_ratio": 1.45}))
