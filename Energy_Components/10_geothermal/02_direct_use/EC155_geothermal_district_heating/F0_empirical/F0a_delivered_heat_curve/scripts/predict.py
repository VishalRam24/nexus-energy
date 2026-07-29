"""Standard predict interface for EC155 F0a delivered-heat curve model."""
import json
import os

import numpy as np

from model import DeliveredHeatCurve

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARAMS = os.path.join(_HERE, "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC155"
    component_name = "Geothermal District Heating"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.p = json.load(f)
        c = self.p["curve"]
        self.model = DeliveredHeatCurve(
            T_source_degC=c["T_source_degC"],
            q_specific=c["q_specific_kW_per_kgps"],
            T_return_ref=self.p["T_return_ref"]["value"],
            cp=self.p["cp_geo"]["value"],
            eta_hx=self.p["heat_transfer_efficiency"]["value"],
            dist_loss=self.p["distribution_losses"]["value"],
            pump_frac=self.p["pump_power_fraction"]["value"],
        )

    def predict(self, inputs: dict) -> dict:
        """inputs: T_source [degC], flow_rate_kgs [kg/s], optional T_return [degC]."""
        T_src = inputs["T_source"]
        m_dot = inputs.get("flow_rate_kgs", self.p["rated"]["m_dot_geo_design"]["value"])
        T_ret = inputs.get("T_return", None)
        q_spec = self.model.q_specific(T_src, T_ret)
        Q = self.model.delivered_heat_kW(T_src, m_dot, T_ret)
        pump = self.model.pump_power_kW(Q)
        return {
            "q_specific_kW_per_kgps": float(np.asarray(q_spec)),
            "Q_delivered_kW": float(np.asarray(Q)),
            "pump_power_kW": float(np.asarray(pump)),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_source": "degC",
                "flow_rate_kgs": "kg/s",
                "T_return": "degC (optional)",
            },
            "outputs": {
                "q_specific_kW_per_kgps": "kW per kg/s",
                "Q_delivered_kW": "kW",
                "pump_power_kW": "kW",
            },
            "valid_ranges": self.p["valid_ranges"],
            "source": self.p["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    sample = {"T_source": 80.0, "flow_rate_kgs": 50.0}
    print("get_info:", json.dumps(m.get_info(), indent=2))
    print("predict", sample, "->", m.predict(sample))
