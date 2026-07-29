"""EC147 -- HTL -- F1b Feedstock Variation -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import HTLF1b


class ComponentModel:
    """Standardized interface for EC147 HTL -- F1b feedstock variation model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = HTLF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "feedstock_type":    str
                "temperature_degC":  float [250-400] (default 330)
                "moisture_fraction": float [0-0.90] (default 0.70)
                "PLR":               float [0.2-1.0] (default 1.0)
                "feed_rate_kg_h":    float (default 500)
            }
        """
        return self._model.predict(
            feedstock_type=str(inputs["feedstock_type"]),
            temperature_degC=float(inputs.get("temperature_degC", 330.0)),
            moisture_fraction=float(inputs.get("moisture_fraction", 0.70)),
            PLR=float(inputs.get("PLR", 1.0)),
            feed_rate_kg_h=float(inputs.get("feed_rate_kg_h", 500.0)),
        )

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Hydrothermal Liquefaction (HTL)",
            "ec_id": "EC147",
            "fidelity": "F1b",
            "model": "Feedstock-Specific Biochemical Composition",
            "description": (
                f"HTL model with {len(m.feedstock_db)} feedstocks. "
                "Lipid/protein/carbohydrate composition drives bio-crude yield. "
                "Temperature peaks at ~330 degC. Moisture-LHV coupling. "
                "Product distribution: bio-crude, aqueous, gas, solid."
            ),
            "inputs": {
                "feedstock_type":    {"type": "str", "options": list(m.feedstock_db.keys())},
                "temperature_degC":  {"unit": "degC", "range": [250.0, 400.0], "default": 330.0},
                "moisture_fraction": {"unit": "—",    "range": [0.0, 0.90],    "default": 0.70},
                "PLR":               {"unit": "—",    "range": [0.20, 1.0],    "default": 1.0},
                "feed_rate_kg_h":    {"unit": "kg/h (wet)"},
            },
            "outputs": {
                "bio_crude_yield":      {"unit": "kg/kg_dry"},
                "product_distribution": {"unit": "kg/kg_dry", "keys": ["bio_crude","aqueous","gas","solid"]},
                "energy_recovery":      {"unit": "dimensionless"},
                "LHV_eff_MJ_kg":        {"unit": "MJ/kg_wet"},
                "moisture_lhv_factor":  {"unit": "dimensionless"},
                "thermal_efficiency":   {"unit": "dimensionless"},
                "bio_crude_rate_kg_h":  {"unit": "kg/h"},
            },
            "source": "Peterson et al. (2008) E&ES; Anastasakis & Ross (2011); Vardon (2011); Elliott (2015)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for fs in ["microalgae_chlorella", "sewage_sludge", "wood_biomass"]:
        r = model.predict({"feedstock_type": fs, "temperature_degC": 330.0, "moisture_fraction": 0.80})
        print(f"{fs}: bc={r['bio_crude_yield']:.3f}, ER={r['energy_recovery']:.3f}")
