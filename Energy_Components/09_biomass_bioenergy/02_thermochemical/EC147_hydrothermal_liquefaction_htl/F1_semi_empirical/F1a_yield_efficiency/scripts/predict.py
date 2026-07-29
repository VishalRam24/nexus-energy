"""EC147 -- Hydrothermal Liquefaction (HTL) -- F1a Yield Model -- Standardized Predict Interface"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import HTLF1a


class ComponentModel:
    component_id = "EC147"

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = HTLF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            feedstock_dry_kg_per_h : float  Dry feedstock feed [kg/h]
        returns:
            bio_crude_kg_per_h, aqueous_kg_per_h, gas_kg_per_h,
            solid_kg_per_h, energy_output_MW, electricity_kW
        """
        feed = float(inputs.get("feedstock_dry_kg_per_h", 1000.0))
        return self._model.predict(feed)

    def get_info(self) -> dict:
        m = self._model
        return {
            "name":        "Hydrothermal Liquefaction (HTL)",
            "ec_id":       "EC147",
            "fidelity":    "F1a",
            "model":       "Yield Model (bio_crude = feedstock_dry * bio_crude_yield)",
            "description": (
                f"HTL at {m.T_operating:.0f} degC / {m.P_operating:.0f} bar. "
                f"bio_crude_yield={m.bio_crude_yield:.0%}, LHV={m.LHV_bio_crude:.0f} MJ/kg. "
                f"Moisture-tolerant process."
            ),
            "inputs":  {"feedstock_dry_kg_per_h": {"unit": "kg/h", "range": [0.0, 1e5]}},
            "outputs": {
                "bio_crude_kg_per_h": {"unit": "kg/h"},
                "aqueous_kg_per_h":   {"unit": "kg/h"},
                "gas_kg_per_h":       {"unit": "kg/h"},
                "solid_kg_per_h":     {"unit": "kg/h"},
                "energy_output_MW":   {"unit": "MW"},
                "electricity_kW":     {"unit": "kW"},
            },
            "source": "Toor et al. (2011) Energy 36:2328; Elliott et al. (2015) BRT 178:147",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")
    r = model.predict({"feedstock_dry_kg_per_h": 1000.0})
    for k, v in r.items():
        print(f"  {k}: {v:.3f}")
