"""Standard ComponentModel interface for EC066 F0a empirical power-curve lookup."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import WindPowerCurve  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC066"
    component_name = "Offshore Floating Wind Turbine (IEA 15 MW)"
    fidelity = "F0a -- empirical lookup"
    version = "1.0.0"

    def __init__(self, data_path=_DATA):
        with open(data_path) as f:
            self.params = json.load(f)
        t = self.params["turbine"]
        pc = self.params["power_curve"]
        self.curve = WindPowerCurve(
            pc["wind_speeds_ms"], pc["power_kw"],
            t["cut_in_speed"]["value"], t["rated_speed"]["value"],
            t["cut_out_speed"]["value"], t["rated_power"]["value"],
            t.get("platform_motion_penalty", {}).get("value", 0.0),
        )

    def predict(self, inputs):
        ws = inputs["wind_speed"]
        p = self.curve.power(ws)
        cf = self.curve.capacity_factor(ws)
        return {"power_kw": p, "capacity_factor": cf}

    def get_info(self):
        t = self.params["turbine"]
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"wind_speed": "m/s"},
            "outputs": {"power_kw": "kW", "capacity_factor": "-"},
            "rated_power_kw": t["rated_power"]["value"],
            "valid_range_wind_speed_ms": [0.0, 30.0],
            "source": self.params["power_curve"]["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print("info:", m.get_info())
    for v in (2.0, 8.0, 25.5):
        print(f"  v={v:5.1f} m/s ->", m.predict({"wind_speed": v}))
