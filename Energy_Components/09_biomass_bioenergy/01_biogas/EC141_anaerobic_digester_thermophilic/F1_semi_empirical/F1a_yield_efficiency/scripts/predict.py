"""EC141 -- Anaerobic Digester (Thermophilic) -- F1a Yield Model -- Standardized Predict Interface"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import AnaerobicDigesterThermophilicF1a


class ComponentModel:
    component_id = "EC141"

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AnaerobicDigesterThermophilicF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            feedstock_VS_kg_per_day : float  Volatile solids input [kgVS/day]
        returns:
            VS_destroyed_kg_per_day, biogas_m3_per_day, CH4_m3_per_day,
            CO2_m3_per_day, energy_kWh_per_day, CH4_fraction
        """
        VS = float(inputs.get("feedstock_VS_kg_per_day", 1000.0))
        return self._model.predict(VS)

    def get_info(self) -> dict:
        m = self._model
        return {
            "name":        "Anaerobic Digester (Thermophilic)",
            "ec_id":       "EC141",
            "fidelity":    "F1a",
            "model":       "Yield Model (biogas = VS_in * VS_destruction * biogas_yield)",
            "description": (
                f"Thermophilic AD at {m.T_operating:.0f} degC, HRT={m.HRT_days:.0f} days. "
                f"biogas_yield={m.biogas_yield:.2f} m3/kgVS, CH4={m.CH4_fraction:.0%}, "
                f"VS_destruction={m.VS_destruction:.0%}."
            ),
            "inputs":  {"feedstock_VS_kg_per_day": {"unit": "kgVS/day", "range": [0.0, 1e6]}},
            "outputs": {
                "VS_destroyed_kg_per_day": {"unit": "kgVS/day"},
                "biogas_m3_per_day":       {"unit": "m3/day"},
                "CH4_m3_per_day":          {"unit": "m3/day"},
                "CO2_m3_per_day":          {"unit": "m3/day"},
                "energy_kWh_per_day":      {"unit": "kWh/day"},
                "CH4_fraction":            {"unit": "dimensionless"},
            },
            "source": "Angelidaki et al. (2009) WST 59:927; Mata-Alvarez et al. (2014) RSE 36:412",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")
    r = model.predict({"feedstock_VS_kg_per_day": 1000.0})
    for k, v in r.items():
        print(f"  {k}: {v:.2f}")
