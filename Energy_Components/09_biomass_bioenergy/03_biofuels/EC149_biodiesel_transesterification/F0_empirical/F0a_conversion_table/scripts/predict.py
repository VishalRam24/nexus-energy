"""Standardized predict interface for EC149 F0a biodiesel conversion table."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import ConversionTable

_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC149"
    component_name = "Biodiesel Transesterification"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_DATA):
        with open(params_path) as f:
            self.params = json.load(f)
        self.table = ConversionTable(self.params)

    def predict(self, inputs: dict) -> dict:
        feedstock = inputs.get("feedstock", "soybean_oil")
        T = float(inputs.get("temperature_degC", self.table.T_opt))
        feed_t = float(inputs.get("feed_tonnes", 1.0))
        ffa = inputs.get("ffa_pct", None)
        conv = self.table.conversion(feedstock=feedstock,
                                      ffa_pct=(float(ffa) if ffa is not None else None), T_degC=T)
        bd_frac = self.table.oil.get(feedstock, 0.2) * conv
        return {
            "conversion": conv,
            "biodiesel_frac": bd_frac,
            "biodiesel_tonnes": bd_frac * feed_t,
            "energy_MJ_per_tonne_feed": bd_frac * 1000.0 * self.table.lhv,
            "ffa_pct": ffa if ffa is not None else self.table.ffa.get(feedstock),
            "temp_multiplier": self.table.temp_multiplier(T),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"feedstock": list(self.table.feedstocks),
                       "temperature_degC": "40-80", "ffa_pct": "0-10", "feed_tonnes": ">0"},
            "outputs": ["conversion", "biodiesel_frac", "biodiesel_tonnes",
                        "energy_MJ_per_tonne_feed"],
            "source": self.params["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print(m.predict({"feedstock": "rapeseed_oil", "temperature_degC": 60, "feed_tonnes": 100}))
