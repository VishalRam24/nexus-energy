"""Standardized prediction interface for the EC215 solar still / HDH F0a yield lookup."""
import json
import os
import numpy as np

try:
    from model import YieldLookup
except ImportError:
    from .model import YieldLookup

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARAMS = os.path.join(_HERE, "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC215"
    component_name = "Solar Still / Humidification-Dehumidification (HDH)"
    fidelity = "F0a - empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.params = json.load(f)
        u = self.params["unit"]
        self.lookup = YieldLookup(
            irr_bp=u["irradiance_breakpoints"]["value"],
            yield_bp=u["yield_breakpoints"]["value"],
            irr_rated=u["irradiance_rated_W_m2"]["value"],
            yield_rated=u["yield_rated_L_m2_day"]["value"],
            collector_area_m2=u["collector_area_m2"]["value"],
            gor_hdh=u["gor_hdh"]["value"],
            gor_min=u["gor_min"]["value"],
            gor_max=u["gor_max"]["value"],
            sec_solar_kWh_m3=u["sec_solar_kWh_m3"]["value"],
        )

    def predict(self, inputs: dict) -> dict:
        irr = inputs.get("solar_irradiance_W_m2", self.lookup.irr_rated)
        area = inputs.get("collector_area_m2", self.lookup.collector_area_m2)
        prod = self.lookup.productivity(irr)
        daily = self.lookup.daily_yield(irr, area)
        return {
            "productivity_L_m2_day": prod,
            "daily_yield_L_day": daily,
            "gor_hdh": np.full_like(np.asarray(prod, dtype=float), self.lookup.gor_hdh),
            "sec_solar_kWh_m3": np.full_like(np.asarray(prod, dtype=float), self.lookup.sec_solar_kWh_m3),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "solar_irradiance_W_m2": "W/m2 (400-1200)",
                "collector_area_m2": "m2",
            },
            "outputs": {
                "productivity_L_m2_day": "L/(m2*day)",
                "daily_yield_L_day": "L/day",
                "gor_hdh": "dimensionless",
                "sec_solar_kWh_m3": "kWh/m3",
            },
            "rated_yield_L_m2_day": self.lookup.yield_rated,
            "source": self.params["unit"]["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print("sample @ 800 W/m2:", m.predict({"solar_irradiance_W_m2": 800.0}))
    print("sample @ 1000 W/m2:", m.predict({"solar_irradiance_W_m2": 1000.0}))
