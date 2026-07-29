"""EC148 -- Bioethanol Fermentation -- F1b Feedstock Variation -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import BioethanolFermentationF1b


class ComponentModel:
    """Standardized interface for EC148 Bioethanol Fermentation -- F1b."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BioethanolFermentationF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "feedstock_type":    str
                "temperature_degC":  float [20-45] (default 32)
                "moisture_fraction": float [0-0.85] (default 0.20)
                "PLR":               float [0.2-1.0] (default 1.0)
                "feed_rate_kg_h":    float (default 1000)
                "ethanol_conc_pct":  float [2-20] (default 8)
            }
        """
        return self._model.predict(
            feedstock_type=str(inputs["feedstock_type"]),
            temperature_degC=float(inputs.get("temperature_degC", 32.0)),
            moisture_fraction=float(inputs.get("moisture_fraction", 0.20)),
            PLR=float(inputs.get("PLR", 1.0)),
            feed_rate_kg_h=float(inputs.get("feed_rate_kg_h", 1000.0)),
            ethanol_conc_pct=float(inputs.get("ethanol_conc_pct", 8.0)),
        )

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Bioethanol Fermentation",
            "ec_id": "EC148",
            "fidelity": "F1b",
            "model": "Feedstock-Specific with Temperature Kinetics",
            "description": (
                f"Bioethanol model with {len(m.feedstock_db)} feedstocks. "
                "Gay-Lussac stoichiometry: 0.511 kg EtOH/kg glucose. "
                "Temperature Arrhenius kinetics, peak at 30-35 degC. "
                "Product inhibition, pretreatment efficiency, moisture-LHV coupling."
            ),
            "inputs": {
                "feedstock_type":   {"type": "str", "options": list(m.feedstock_db.keys())},
                "temperature_degC": {"unit": "degC", "range": [20.0, 45.0],  "default": 32.0},
                "moisture_fraction":{"unit": "—",    "range": [0.0, 0.85],   "default": 0.20},
                "PLR":              {"unit": "—",    "range": [0.20, 1.0],   "default": 1.0},
                "ethanol_conc_pct": {"unit": "%v/v", "range": [2.0, 20.0],   "default": 8.0},
            },
            "outputs": {
                "ethanol_yield":       {"unit": "kg EtOH/kg_dry"},
                "sugar_fraction":      {"unit": "kg/kg_dry"},
                "temperature_factor":  {"unit": "dimensionless"},
                "pretreatment_eff":    {"unit": "dimensionless"},
                "LHV_eff_MJ_kg":       {"unit": "MJ/kg_wet"},
                "moisture_lhv_factor": {"unit": "dimensionless"},
                "thermal_efficiency":  {"unit": "dimensionless"},
                "ethanol_rate_kg_h":   {"unit": "kg/h"},
            },
            "source": "Balat et al. (2008) PECS; Chandel (2011); Wyman (1999)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for fs in ["corn", "sugarcane", "wheat_straw", "switchgrass"]:
        r = model.predict({"feedstock_type": fs, "temperature_degC": 32.0})
        print(f"{fs}: Y={r['ethanol_yield']:.4f} kg/kg, sugar={r['sugar_fraction']:.3f}, f_T={r['temperature_factor']:.3f}")
