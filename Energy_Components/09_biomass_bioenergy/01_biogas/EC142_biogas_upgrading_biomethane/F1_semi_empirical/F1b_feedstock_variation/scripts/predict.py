"""EC142 -- Biogas Upgrading -- F1b -- Standard Predict Interface"""
import json
from pathlib import Path
from model import BiogasUpgradingF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BiogasUpgradingF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        return self._model.predict(
            feedstock_type    = inputs["feedstock_type"],
            biogas_flow_m3_h  = inputs["biogas_flow_m3_h"],
            moisture_fraction = inputs.get("moisture_fraction", 0.0),
            temperature_degC  = inputs.get("temperature_degC", 20.0),
        )

    def get_info(self) -> dict:
        return {
            "name":     "Biogas Upgrading / Biomethane",
            "ec_id":    "EC142",
            "fidelity": "F1b",
            "description": "Feedstock-specific biogas composition, PSA upgrading model, "
                           "moisture-LHV coupling, H2S removal, biomethane spec check",
            "inputs": {
                "feedstock_type":   {"options": list(self._model.feedstock_db.keys())},
                "biogas_flow_m3_h": {"unit": "m3/h", "range": [10.0, 5000.0]},
                "moisture_fraction":{"unit": "-", "range": [0.0, 0.6], "default": 0.0},
            },
            "outputs": {
                "biomethane_flow_m3_h":  {"unit": "m3/h"},
                "methane_recovery_pct":  {"unit": "%"},
                "biomethane_CH4_pct":    {"unit": "%"},
                "CO2_removal_pct":       {"unit": "%"},
                "H2S_product_ppm":       {"unit": "ppm"},
                "upgrading_energy_kwh_h":{"unit": "kWh/h"},
                "net_energy_kwh_h":      {"unit": "kWh/h"},
                "meets_spec":            {"unit": "bool"},
                "moisture_lhv_factor":   {"unit": "-"},
            },
            "source": "Bauer et al. (2013); Ryckebosch et al. (2011); IEA Bioenergy Task 37",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"feedstock_type": "food_waste", "biogas_flow_m3_h": 100.0})
    print(json.dumps(r, indent=2))
