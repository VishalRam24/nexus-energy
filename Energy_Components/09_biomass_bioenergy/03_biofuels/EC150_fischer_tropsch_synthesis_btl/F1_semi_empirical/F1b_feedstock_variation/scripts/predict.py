"""EC150 -- Fischer-Tropsch (BtL) -- F1b Feedstock Variation -- Standardized Interface"""
import json, numpy as np
from pathlib import Path
from model import FischerTropschF1b


class ComponentModel:
    """Standardized interface for EC150 Fischer-Tropsch BtL -- F1b."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = FischerTropschF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        return self._model.predict(
            feedstock_type=str(inputs["feedstock_type"]),
            temperature_degC=float(inputs.get("temperature_degC", 230.0)),
            moisture_fraction=float(inputs.get("moisture_fraction", 0.10)),
            PLR=float(inputs.get("PLR", 1.0)),
            feed_rate_kg_h=float(inputs.get("feed_rate_kg_h", 1000.0)),
        )

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Fischer-Tropsch Synthesis (BtL)",
            "ec_id": "EC150",
            "fidelity": "F1b",
            "model": "ASF Chain Growth + Feedstock-Specific Syngas Quality",
            "inputs": {
                "feedstock_type":   {"type": "str", "options": list(m.feedstock_db.keys())},
                "temperature_degC": {"unit": "degC", "range": [180.0, 350.0]},
                "moisture_fraction":{"unit": "—",    "range": [0.0, 0.60]},
                "PLR":              {"unit": "—",    "range": [0.30, 1.0]},
            },
            "outputs": {
                "ft_liquid_yield":     {"unit": "kg/kg_dry"},
                "product_selectivity": {"unit": "mass fractions", "keys": ["CH4","LPG","gasoline","diesel","wax"]},
                "alpha":               {"unit": "dimensionless"},
                "co_conversion":       {"unit": "dimensionless"},
                "h2_co_ratio":         {"unit": "mol/mol"},
                "LHV_eff_MJ_kg":       {"unit": "MJ/kg_wet"},
                "thermal_efficiency":  {"unit": "dimensionless"},
                "ft_liquid_rate_kg_h": {"unit": "kg/h"},
                "diesel_rate_kg_h":    {"unit": "kg/h"},
            },
            "source": "Anderson (1984); Dry (2002) Catalysis Today; van der Laan & Beenackers (1999)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for fs in ["wood_chips", "torrefied_wood", "municipal_solid_waste"]:
        r = model.predict({"feedstock_type": fs, "temperature_degC": 230.0})
        print(f"{fs}: Y_liq={r['ft_liquid_yield']:.3f}, alpha={r['alpha']:.3f}, H2/CO={r['h2_co_ratio']:.2f}")
        print(f"  diesel={r['product_selectivity']['diesel']:.3f}, wax={r['product_selectivity']['wax']:.3f}")
