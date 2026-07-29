"""EC149 -- Biodiesel Transesterification -- F1b Feedstock Variation -- Standardized Interface"""
import json, numpy as np
from pathlib import Path
from model import BiodieselTransesterificationF1b


class ComponentModel:
    """Standardized interface for EC149 Biodiesel Transesterification -- F1b."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BiodieselTransesterificationF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        return self._model.predict(
            feedstock_type=str(inputs["feedstock_type"]),
            temperature_degC=float(inputs.get("temperature_degC", 60.0)),
            moisture_fraction=float(inputs.get("moisture_fraction", 0.05)),
            PLR=float(inputs.get("PLR", 1.0)),
            feed_rate_kg_h=float(inputs.get("feed_rate_kg_h", 1000.0)),
            ffa_override=inputs.get("ffa_pct", None),
        )

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Biodiesel Transesterification",
            "ec_id": "EC149",
            "fidelity": "F1b",
            "model": "Feedstock-Specific with FFA and Temperature Effects",
            "inputs": {
                "feedstock_type":   {"type": "str", "options": list(m.feedstock_db.keys())},
                "temperature_degC": {"unit": "degC", "range": [40.0, 80.0]},
                "moisture_fraction":{"unit": "—",    "range": [0.0, 0.30]},
                "PLR":              {"unit": "—",    "range": [0.20, 1.0]},
                "ffa_pct":          {"unit": "%",    "note": "Optional override"},
            },
            "outputs": {
                "biodiesel_yield":    {"unit": "kg FAME/kg_raw"},
                "glycerol_yield":     {"unit": "kg/kg_raw"},
                "temperature_factor": {"unit": "dimensionless"},
                "ffa_penalty_factor": {"unit": "dimensionless"},
                "LHV_eff_MJ_kg":      {"unit": "MJ/kg_wet"},
                "thermal_efficiency": {"unit": "dimensionless"},
                "biodiesel_rate_kg_h":{"unit": "kg/h"},
                "glycerol_rate_kg_h": {"unit": "kg/h"},
            },
            "source": "Freedman (1984); Rashid (2008); Meher (2006); Ma & Hanna (1999)",
            "ec_id": "EC149",
            "fidelity": "F1b",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for fs in ["soybean_oil", "rapeseed_oil", "waste_cooking_oil", "jatropha"]:
        r = model.predict({"feedstock_type": fs, "temperature_degC": 60.0})
        print(f"{fs}: FAME={r['biodiesel_yield']:.3f}, ffa_factor={r['ffa_penalty_factor']:.3f}, oil={r['oil_content']:.2f}")
