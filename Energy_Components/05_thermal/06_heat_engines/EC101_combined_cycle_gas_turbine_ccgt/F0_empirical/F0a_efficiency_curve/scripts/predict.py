"""Standard predict interface for EC101 Combined Cycle Gas Turbine (CCGT) (F0a empirical lookup)."""
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARAMS = os.path.join(_HERE, "..", "data", "parameters.json")

try:
    from model import EfficiencyCurve
except ImportError:
    import sys
    sys.path.insert(0, _HERE)
    from model import EfficiencyCurve


class ComponentModel:
    component_id = "EC101"
    component_name = "Combined Cycle Gas Turbine (CCGT)"
    fidelity = "F0a - empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.p = json.load(f)
        lk = self.p["lookup"]
        rt = self.p["rated"]
        self.curve = EfficiencyCurve(
            lk["plr_breakpoints"]["value"],
            lk["eta_breakpoints"]["value"],
            rt["P_rated"]["value"],
            rt["PLR_min"]["value"],
        )

    def predict(self, inputs: dict) -> dict:
        plr = inputs.get("part_load_ratio", inputs.get("plr", 1.0))
        eta = self.curve.efficiency(plr)
        p_out = self.curve.power_out(plr)
        q_in = self.curve.fuel_heat_in(plr)
        return {
            "electrical_efficiency": float(np.asarray(eta).item()) if np.ndim(eta) == 0 else np.asarray(eta),
            "power_output_W": float(np.asarray(p_out).item()) if np.ndim(p_out) == 0 else np.asarray(p_out),
            "fuel_heat_input_W": float(np.asarray(q_in).item()) if np.ndim(q_in) == 0 else np.asarray(q_in),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"part_load_ratio": "dimensionless [0-1]"},
            "outputs": {
                "electrical_efficiency": "dimensionless",
                "power_output_W": "W",
                "fuel_heat_input_W": "W",
            },
            "rated_power_W": self.p["rated"]["P_rated"]["value"],
            "eta_rated": self.p["rated"]["eta_rated"]["value"],
            "source": self.p["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.component_id, m.component_name, "|", m.fidelity, "v" + m.version)
    for plr in (1.0, 0.5, 0.1):
        print(f"PLR={plr:.2f} ->", m.predict({"part_load_ratio": plr}))
