"""EC145 -- Pyrolysis Reactor -- F1a Yield Model -- Standardized Predict Interface"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import PyrolysisReactorF1a


class ComponentModel:
    component_id = "EC145"

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PyrolysisReactorF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            feedstock_dry_kg_per_h : float  Dry biomass feed [kg/h]
        returns:
            bio_oil_kg_per_h, char_kg_per_h, gas_kg_per_h,
            energy_bio_oil_MW, energy_char_MW, energy_gas_MW, mass_balance_check
        """
        feed = float(inputs.get("feedstock_dry_kg_per_h", 1000.0))
        return self._model.predict(feed)

    def get_info(self) -> dict:
        m = self._model
        return {
            "name":        "Pyrolysis Reactor (Fast Pyrolysis)",
            "ec_id":       "EC145",
            "fidelity":    "F1a",
            "model":       "Yield Model (product = feedstock * yield_fraction)",
            "description": (
                f"Fast pyrolysis at {m.T_operating:.0f} degC. "
                f"bio_oil={m.bio_oil_yield:.0%}, char={m.char_yield:.0%}, gas={m.gas_yield:.0%}. "
                f"LHV_bio_oil={m.LHV_bio_oil:.0f} MJ/kg."
            ),
            "inputs":  {"feedstock_dry_kg_per_h": {"unit": "kg/h", "range": [0.0, 1e5]}},
            "outputs": {
                "bio_oil_kg_per_h":  {"unit": "kg/h"},
                "char_kg_per_h":     {"unit": "kg/h"},
                "gas_kg_per_h":      {"unit": "kg/h"},
                "energy_bio_oil_MW": {"unit": "MW"},
                "energy_char_MW":    {"unit": "MW"},
                "energy_gas_MW":     {"unit": "MW"},
                "mass_balance_check":{"unit": "dimensionless"},
            },
            "source": "Bridgwater (2012) BB 38:68; Mohan et al. (2006) EF 20:848",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")
    r = model.predict({"feedstock_dry_kg_per_h": 1000.0})
    for k, v in r.items():
        print(f"  {k}: {v:.3f}")
