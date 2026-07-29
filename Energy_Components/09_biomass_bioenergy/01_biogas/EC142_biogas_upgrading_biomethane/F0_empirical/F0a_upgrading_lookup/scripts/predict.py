"""Standardized predict interface for EC142 F0a upgrading lookup."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import UpgradingLookup

_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC142"
    component_name = "Biogas Upgrading / Biomethane"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_DATA):
        with open(params_path) as f:
            self.params = json.load(f)
        self.table = UpgradingLookup(self.params)

    def predict(self, inputs: dict) -> dict:
        flow = float(inputs.get("biogas_flow_m3_h", 100.0))
        if "raw_CH4_pct" in inputs:
            raw = float(inputs["raw_CH4_pct"])
        else:
            raw = self.table.raw_ch4_pct(inputs.get("feedstock", "cattle_manure"))
        bm_ratio = self.table.biomethane_per_biogas(raw)
        bm_flow = bm_ratio * flow
        return {
            "raw_CH4_pct": raw,
            "ch4_recovery": self.table.recovery,
            "product_CH4_pct": self.table.product_purity,
            "biomethane_m3_h": bm_flow,
            "biomethane_per_biogas": bm_ratio,
            "upgrading_power_kw": self.table.upg_energy * flow,
            "parasitic_fraction": self.table.parasitic_fraction(raw),
            "energy_out_kw": bm_flow * self.table.lhv,
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"feedstock": list(self.table.feedstocks),
                       "raw_CH4_pct": "50-70", "biogas_flow_m3_h": "10-5000"},
            "outputs": ["biomethane_m3_h", "ch4_recovery", "upgrading_power_kw",
                        "parasitic_fraction", "energy_out_kw"],
            "source": self.params["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print(m.predict({"feedstock": "food_waste", "biogas_flow_m3_h": 500}))
