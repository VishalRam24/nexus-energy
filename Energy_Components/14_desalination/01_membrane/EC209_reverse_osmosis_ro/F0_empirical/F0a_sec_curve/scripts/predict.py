"""Standardized prediction interface for the EC209 RO F0a SEC-vs-recovery curve."""
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
    component_id = "EC209"
    component_name = "Reverse Osmosis (RO)"
    fidelity = "F0a - empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.params = json.load(f)
        u = self.params["unit"]
        self.S_feed = u["S_feed"]["value"]
        self.curve = SECCurve(
            recovery_bp=u["recovery_breakpoints"]["value"],
            sec_bp=u["sec_breakpoints"]["value"],
            recovery_rated=u["recovery_rated"]["value"],
            sec_rated=u["sec_rated"]["value"],
            rejection=u["permeate_salinity_rejection"]["value"],
        )

    def predict(self, inputs: dict) -> dict:
        recovery = inputs.get("recovery", self.curve.recovery_rated)
        feed_salinity = inputs.get("feed_salinity", self.S_feed)
        feed_flow = inputs.get("feed_flow_m3h", 100.0)
        sec = self.curve.sec(recovery)
        permeate_flow = np.asarray(feed_flow, dtype=float) * np.asarray(recovery, dtype=float)
        power = sec * permeate_flow
        return {
            "sec_kWh_m3": sec,
            "permeate_flow_m3h": permeate_flow,
            "power_kW": power,
            "permeate_salinity_g_L": self.curve.permeate_salinity(recovery, feed_salinity),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "recovery": "dimensionless (0.2-0.6)",
                "feed_salinity": "g/L",
                "feed_flow_m3h": "m3/hr",
            },
            "outputs": {
                "sec_kWh_m3": "kWh/m3",
                "permeate_flow_m3h": "m3/hr",
                "power_kW": "kW",
                "permeate_salinity_g_L": "g/L",
            },
            "rated_sec_kWh_m3": self.curve.sec_rated,
            "source": self.params["unit"]["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print("sample @ recovery=0.45:", m.predict({"recovery": 0.45}))
    print("sample @ recovery=0.55:", m.predict({"recovery": 0.55}))
