"""Standard predict interface for EC152 F0a efficiency-curve model."""
import json
import os

import numpy as np

from model import EfficiencyCurve

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARAMS = os.path.join(_HERE, "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC152"
    component_name = "Flash Steam Geothermal Plant"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.p = json.load(f)
        c = self.p["curve"]
        self.model = EfficiencyCurve(
            T_geo_degC=c["T_geo_degC"],
            eta_net=c["eta_net"],
            T_reject_ref=self.p["T_reject_ref"]["value"],
            cp=self.p["valid_ranges"]["cp_steam"]["value"],
            eta_utilization=self.p["eta_utilization"]["value"],
        )

    def predict(self, inputs: dict) -> dict:
        """inputs: T_geothermal [degC], flow_rate_kgs [kg/s], optional T_rejection [degC]."""
        T_geo = inputs["T_geothermal"]
        m_dot = inputs.get("flow_rate_kgs", self.p["valid_ranges"]["flow_rate_kgs"]["min"])
        T_rej = inputs.get("T_rejection", None)
        eta = self.model.eta_net(T_geo)
        p_net = self.model.net_power_kW(T_geo, m_dot, T_rej)
        return {
            "eta_net": float(np.asarray(eta)),
            "net_power_kW": float(np.asarray(p_net)),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_geothermal": "degC",
                "flow_rate_kgs": "kg/s",
                "T_rejection": "degC (optional)",
            },
            "outputs": {"eta_net": "dimensionless", "net_power_kW": "kW"},
            "valid_ranges": self.p["valid_ranges"],
            "source": self.p["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    sample = {"T_geothermal": 200.0, "flow_rate_kgs": 50.0}
    print("get_info:", json.dumps(m.get_info(), indent=2))
    print("predict", sample, "->", m.predict(sample))
