"""F0a empirical predict interface for EC186 STATCOM."""
import json
import os

import numpy as np

try:
    from model import LookupCurve
except ImportError:
    from .model import LookupCurve

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARAMS = os.path.join(_HERE, "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC186"
    component_name = "STATCOM"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.p = json.load(f)
        self.q_max = self.p["Q_max"]["value"]
        self.q_min = self.p["Q_min"]["value"]
        self.loss_factor = self.p["loss_factor"]["value"]
        self.curve = LookupCurve(
            self.p["q_fraction_breakpoints"]["value"],
            self.p["loss_fraction_breakpoints"]["value"],
        )

    def predict(self, inputs: dict) -> dict:
        """inputs: {q_demand}. Returns delivered Q (clamped) and converter losses."""
        q_dem = float(inputs.get("q_demand", self.q_max))
        q_out = float(np.clip(q_dem, self.q_min, self.q_max))
        q_frac = abs(q_out) / self.q_max if self.q_max else 0.0
        loss_frac = float(self.curve.lookup(q_frac))
        loss = loss_frac * abs(q_out)
        return {"q_out": q_out, "q_clamped": bool(q_out != q_dem),
                "loss_fraction": loss_frac, "loss": loss}

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"q_demand": "requested reactive output (MVAR)"},
            "outputs": {"q_out": "MVAR (clamped to [Q_min, Q_max])",
                        "q_clamped": "bool", "loss_fraction": "dimensionless",
                        "loss": "MVAR (active loss)"},
            "Q_max": self.q_max, "Q_min": self.q_min, "loss_factor": self.loss_factor,
            "source": self.p["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print("get_info:", json.dumps(m.get_info(), indent=2))
    print("full cap:", m.predict({"q_demand": m.q_max}))
    print("over-demand:", m.predict({"q_demand": m.q_max * 2}))
