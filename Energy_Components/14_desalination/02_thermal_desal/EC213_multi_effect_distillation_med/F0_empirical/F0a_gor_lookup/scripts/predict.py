"""Standardized prediction interface for the EC213 MED F0a GOR lookup."""
import json
import os
import numpy as np

try:
    from model import GORLookup
except ImportError:
    from .model import GORLookup

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARAMS = os.path.join(_HERE, "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC213"
    component_name = "Multi-Effect Distillation (MED)"
    fidelity = "F0a - empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.params = json.load(f)
        u = self.params["unit"]
        self.lookup = GORLookup(
            T_top_bp=u["T_top_breakpoints"]["value"],
            gor_bp=u["gor_breakpoints"]["value"],
            T_top_rated=u["T_top_rated_C"]["value"],
            gor_rated=u["gor_rated"]["value"],
            sec_thermal_kJ_kg=u["sec_thermal_kJ_kg"]["value"],
            sec_elec_kWh_m3=u["sec_elec_kWh_m3"]["value"],
            recovery=u["recovery"]["value"],
            capacity_m3_h=u["capacity_m3_h"]["value"],
            h_latent_kJ_kg=u["h_latent_kJ_kg"]["value"],
        )

    def predict(self, inputs: dict) -> dict:
        T_top = inputs.get("T_top_C", self.lookup.T_top_rated)
        load = inputs.get("capacity_fraction", 1.0)
        gor = self.lookup.gor(T_top)
        dist = self.lookup.distillate_flow(load)
        return {
            "gor": gor,
            "thermal_sec_kJ_kg": self.lookup.thermal_sec_from_gor(T_top),
            "sec_elec_kWh_m3": np.full_like(np.asarray(gor, dtype=float), self.lookup.sec_elec_kWh_m3),
            "distillate_flow_m3h": dist,
            "elec_power_kW": self.lookup.sec_elec_kWh_m3 * dist,
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_top_C": "degC (55-85)",
                "capacity_fraction": "dimensionless (0.1-1.0)",
            },
            "outputs": {
                "gor": "dimensionless",
                "thermal_sec_kJ_kg": "kJ/kg",
                "sec_elec_kWh_m3": "kWh/m3",
                "distillate_flow_m3h": "m3/hr",
                "elec_power_kW": "kW",
            },
            "rated_gor": self.lookup.gor_rated,
            "source": self.params["unit"]["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print("sample @ T_top=70C:", m.predict({"T_top_C": 70.0}))
    print("sample @ T_top=60C:", m.predict({"T_top_C": 60.0}))
