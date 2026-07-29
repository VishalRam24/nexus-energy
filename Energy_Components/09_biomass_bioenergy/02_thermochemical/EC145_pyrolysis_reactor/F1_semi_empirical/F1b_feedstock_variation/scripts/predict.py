"""EC145 -- Pyrolysis Reactor -- F1b Feedstock Variation -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import PyrolysisReactorF1b


class ComponentModel:
    """Standardized interface for EC145 Pyrolysis Reactor -- F1b feedstock variation model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PyrolysisReactorF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "feedstock_type":    str, e.g. "wood_chips", "pine", "corn_stover"
                "temperature_degC":  float [300-700]
                "moisture_fraction": float [0-0.60] (default 0.10)
                "PLR":               float [0.2-1.0] (default 1.0)
                "feed_rate_kg_h":    float (default 500)
            }
        Returns:
            dict with: bio_oil_yield, char_yield, gas_yield, LHV_eff_MJ_kg,
                       moisture_lhv_factor, energy_recovery, thermal_efficiency,
                       bio_oil_rate_kg_h, char_rate_kg_h, gas_rate_kg_h
        """
        return self._model.predict(
            feedstock_type=str(inputs["feedstock_type"]),
            temperature_degC=float(inputs.get("temperature_degC", 500.0)),
            moisture_fraction=float(inputs.get("moisture_fraction", 0.10)),
            PLR=float(inputs.get("PLR", 1.0)),
            feed_rate_kg_h=float(inputs.get("feed_rate_kg_h", 500.0)),
        )

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Pyrolysis Reactor",
            "ec_id": "EC145",
            "fidelity": "F1b",
            "model": "Feedstock-Specific with Moisture-LHV Coupling",
            "description": (
                f"Feedstock-specific pyrolysis model with {len(m.feedstock_db)} feedstocks. "
                "Moisture-LHV coupling: LHV_eff = LHV_dry*(1-M) - h_fg*M. "
                "Temperature-dependent bio-oil/char/gas yields with 500 degC peak for bio-oil. "
                "Part-load thermal efficiency polynomial."
            ),
            "inputs": {
                "feedstock_type":   {"type": "str", "options": list(m.feedstock_db.keys())},
                "temperature_degC": {"unit": "degC",          "range": [300.0, 700.0]},
                "moisture_fraction":{"unit": "dimensionless", "range": [0.0, 0.60], "default": 0.10},
                "PLR":              {"unit": "dimensionless", "range": [0.20, 1.0],  "default": 1.0},
                "feed_rate_kg_h":   {"unit": "kg/h",          "range": [10.0, 5000.0]},
            },
            "outputs": {
                "bio_oil_yield":       {"unit": "kg/kg_dry"},
                "char_yield":          {"unit": "kg/kg_dry"},
                "gas_yield":           {"unit": "kg/kg_dry"},
                "LHV_eff_MJ_kg":       {"unit": "MJ/kg_wet"},
                "moisture_lhv_factor": {"unit": "dimensionless"},
                "energy_recovery":     {"unit": "dimensionless"},
                "thermal_efficiency":  {"unit": "dimensionless"},
                "bio_oil_rate_kg_h":   {"unit": "kg/h"},
                "char_rate_kg_h":      {"unit": "kg/h"},
                "gas_rate_kg_h":       {"unit": "kg/h"},
            },
            "source": "Bridgwater (2012) Biomass & Bioenergy; Demirbas (2004); Jenkins et al. (1998)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")
    for fs in ["wood_chips", "pine", "corn_stover"]:
        r = model.predict({"feedstock_type": fs, "temperature_degC": 500.0, "moisture_fraction": 0.10})
        print(f"\n{fs} (500°C, M=10%):")
        print(f"  bio_oil={r['bio_oil_yield']:.3f}  char={r['char_yield']:.3f}  gas={r['gas_yield']:.3f}")
        print(f"  LHV_eff={r['LHV_eff_MJ_kg']:.2f} MJ/kg  recovery={r['energy_recovery']:.3f}  eta_th={r['thermal_efficiency']:.3f}")
