"""EC141 -- Anaerobic Digester (Thermophilic) -- F1b -- Standard Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import AnaerobicDigesterThermophilicF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AnaerobicDigesterThermophilicF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
          feedstock_type         : str or dict of {name: fraction}
          vs_loading_kg_m3_day   : float [kgVS/(m3*day)]
          temperature_degC       : float [degC] (default 55)
          hrt_days               : float [days] (default 15)
          moisture_fraction      : float [0-1] (default 0)
        returns:
          biogas_yield_m3_day, methane_content_pct, methane_yield_m3_day,
          vs_removal_pct, cn_ratio, moisture_lhv_factor
        """
        return self._model.predict(
            feedstock_type        = inputs["feedstock_type"],
            vs_loading_kg_m3_day  = inputs["vs_loading_kg_m3_day"],
            temperature_degC      = inputs.get("temperature_degC", 55.0),
            hrt_days              = inputs.get("hrt_days", 15.0),
            moisture_fraction     = inputs.get("moisture_fraction", 0.0),
        )

    def get_info(self) -> dict:
        return {
            "name":        "Anaerobic Digester (Thermophilic)",
            "ec_id":       "EC141",
            "fidelity":    "F1b",
            "description": "Feedstock-specific BMP, co-digestion synergy, C/N penalty, "
                           "thermophilic temperature model, moisture-LHV coupling",
            "inputs": {
                "feedstock_type":       {"options": list(self._model.feedstock_db.keys())},
                "vs_loading_kg_m3_day": {"unit": "kgVS/(m3*day)", "range": [0.5, 12.0]},
                "temperature_degC":     {"unit": "degC", "range": [45.0, 65.0], "default": 55.0},
                "hrt_days":             {"unit": "days", "range": [5.0, 30.0], "default": 15.0},
                "moisture_fraction":    {"unit": "-", "range": [0.0, 0.6], "default": 0.0},
            },
            "outputs": {
                "biogas_yield_m3_day":  {"unit": "m3/day"},
                "methane_content_pct":  {"unit": "%"},
                "methane_yield_m3_day": {"unit": "m3_CH4/day"},
                "vs_removal_pct":       {"unit": "%"},
                "cn_ratio":             {"unit": "-"},
                "moisture_lhv_factor":  {"unit": "-"},
            },
            "source": "Angelidaki et al. (2009); Labatut et al. (2011); Mata-Alvarez et al. (2014)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"feedstock_type": "food_waste", "vs_loading_kg_m3_day": 3.0})
    print(json.dumps(r, indent=2))
