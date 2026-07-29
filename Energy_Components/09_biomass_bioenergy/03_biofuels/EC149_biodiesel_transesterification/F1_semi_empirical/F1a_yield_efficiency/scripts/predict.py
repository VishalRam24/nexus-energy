"""EC149 -- Biodiesel Transesterification -- F1a Yield Model -- Standardized Predict Interface"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import BiodieselTransesterificationF1a


class ComponentModel:
    component_id = "EC149"

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BiodieselTransesterificationF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            oil_input_kg_per_h : float  Vegetable oil / waste oil feed [kg/h]
        returns:
            FAME_kg_per_h, FAME_L_per_h, glycerol_kg_per_h,
            methanol_consumed_kg_per_h, energy_output_MW, FAME_yield
        """
        oil = float(inputs.get("oil_input_kg_per_h", 1000.0))
        return self._model.predict(oil)

    def get_info(self) -> dict:
        m = self._model
        return {
            "name":        "Biodiesel Transesterification",
            "ec_id":       "EC149",
            "fidelity":    "F1a",
            "model":       "Yield Model (FAME = oil * FAME_yield; methanol 6:1 molar ratio)",
            "description": (
                f"Alkali-catalyzed transesterification at {m.T_reaction:.0f} degC. "
                f"FAME_yield={m.FAME_yield:.0%}, methanol_ratio=6:1 mol/mol. "
                f"LHV_FAME={m.LHV_FAME:.1f} MJ/kg."
            ),
            "inputs":  {"oil_input_kg_per_h": {"unit": "kg/h", "range": [0.0, 1e5]}},
            "outputs": {
                "FAME_kg_per_h":              {"unit": "kg/h"},
                "FAME_L_per_h":               {"unit": "L/h"},
                "glycerol_kg_per_h":          {"unit": "kg/h"},
                "methanol_consumed_kg_per_h": {"unit": "kg/h"},
                "energy_output_MW":           {"unit": "MW"},
                "FAME_yield":                 {"unit": "dimensionless"},
            },
            "source": "Freedman et al. (1984) JAOCS 61:1638; van Gerpen (2005) FPT 86:1097",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")
    r = model.predict({"oil_input_kg_per_h": 1000.0})
    for k, v in r.items():
        print(f"  {k}: {v:.3f}")
