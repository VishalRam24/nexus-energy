"""EC140 -- Anaerobic Digester -- F1b Feedstock Variation -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import AnaerobicDigesterF1b


class ComponentModel:
    """Standardized interface for EC140 Anaerobic Digester -- F1b feedstock model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AnaerobicDigesterF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "feedstock_type":        str or dict, e.g. "cattle_manure" or
                                         {"cattle_manure": 0.6, "food_waste": 0.4}
                "vs_loading_kg_m3_day":  float [0.5-10.0]
                "temperature_degC":      float [25-45] (default 37)
                "hrt_days":              float [5-60] (default 20)
            }

        Returns:
            dict with:
                biogas_yield_m3_day   : float
                methane_content_pct   : float
                methane_yield_m3_day  : float
                vs_removal_pct        : float
                cn_ratio              : float
        """
        return self._model.predict(
            feedstock_type=inputs["feedstock_type"],
            vs_loading_kg_m3_day=float(inputs.get("vs_loading_kg_m3_day", 3.0)),
            temperature_degC=float(inputs.get("temperature_degC", 37.0)),
            hrt_days=float(inputs.get("hrt_days", 20.0)),
        )

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Anaerobic Digester (Mesophilic)",
            "ec_id": "EC140",
            "fidelity": "F1b",
            "model": "Feedstock-Specific BMP with Co-Digestion",
            "description": (
                f"Feedstock-specific BMP model with {len(m.feedstock_db)} feedstocks. "
                "Co-digestion synergy up to 10% when C/N ratio in [20,30]. "
                f"Digester volume: {m.V:.0f} m3. "
                "Temperature correction via Arrhenius. "
                "First-order hydrolysis kinetics for VS removal."
            ),
            "inputs": {
                "feedstock_type":       {"type": "str or dict", "options": list(m.feedstock_db.keys())},
                "vs_loading_kg_m3_day": {"unit": "kgVS/(m3*day)", "range": [0.5, 10.0]},
                "temperature_degC":     {"unit": "degC", "range": [25.0, 45.0], "default": 37.0},
                "hrt_days":             {"unit": "days", "range": [5.0, 60.0], "default": 20.0},
            },
            "outputs": {
                "biogas_yield_m3_day":  {"unit": "m3/day"},
                "methane_content_pct":  {"unit": "%"},
                "methane_yield_m3_day": {"unit": "m3_CH4/day"},
                "vs_removal_pct":       {"unit": "%"},
                "cn_ratio":             {"unit": "dimensionless"},
            },
            "source": "Angelidaki et al. (2009); Mata-Alvarez et al. (2014)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")

    # Single feedstock
    r = model.predict({
        "feedstock_type": "food_waste",
        "vs_loading_kg_m3_day": 3.0,
        "temperature_degC": 37.0,
        "hrt_days": 20.0,
    })
    print(f"\nFood waste (single):")
    for k, v in r.items():
        print(f"  {k}: {v:.2f}")

    # Co-digestion blend
    r2 = model.predict({
        "feedstock_type": {"cattle_manure": 0.6, "food_waste": 0.4},
        "vs_loading_kg_m3_day": 3.0,
        "temperature_degC": 37.0,
        "hrt_days": 20.0,
    })
    print(f"\nCo-digestion (60% manure + 40% food waste):")
    for k, v in r2.items():
        print(f"  {k}: {v:.2f}")
