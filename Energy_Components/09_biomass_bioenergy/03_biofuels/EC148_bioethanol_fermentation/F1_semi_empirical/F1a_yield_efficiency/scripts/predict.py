"""EC148 -- Bioethanol Fermentation -- F1a Yield Model -- Standardized Predict Interface"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import BioethanolFermentationF1a


class ComponentModel:
    component_id = "EC148"

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BioethanolFermentationF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            sugar_input_kg_per_h : float  Fermentable sugars [kg/h]
        returns:
            ethanol_L_per_h, ethanol_kg_per_h, CO2_kg_per_h,
            energy_output_MW, eta_conversion
        """
        sugar = float(inputs.get("sugar_input_kg_per_h", 1000.0))
        return self._model.predict(sugar)

    def get_info(self) -> dict:
        m = self._model
        return {
            "name":        "Bioethanol Fermentation",
            "ec_id":       "EC148",
            "fidelity":    "F1a",
            "model":       "Yield Model (ethanol = sugar * yield * eta_conversion)",
            "description": (
                f"Saccharomyces cerevisiae fermentation at {m.T_ferment:.0f} degC. "
                f"yield={m.ethanol_yield:.2f} L/kg_sugar, eta_conversion={m.eta_conversion:.0%}. "
                f"LHV_ethanol={m.LHV_ethanol:.1f} MJ/kg."
            ),
            "inputs":  {"sugar_input_kg_per_h": {"unit": "kg/h", "range": [0.0, 1e5]}},
            "outputs": {
                "ethanol_L_per_h":  {"unit": "L/h"},
                "ethanol_kg_per_h": {"unit": "kg/h"},
                "CO2_kg_per_h":     {"unit": "kg/h"},
                "energy_output_MW": {"unit": "MW"},
                "eta_conversion":   {"unit": "dimensionless"},
            },
            "source": "Balat (2011) ECM 52:858; Mosier et al. (2005) BRT 96:673",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")
    r = model.predict({"sugar_input_kg_per_h": 1000.0})
    for k, v in r.items():
        print(f"  {k}: {v:.3f}")
