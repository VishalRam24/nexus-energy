"""F0a empirical predict interface for EC183 Circuit Breaker."""
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
    component_id = "EC183"
    component_name = "Circuit Breaker"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.p = json.load(f)
        self.p_rated = self.p["p_rated"]["value"]
        self.loss_rated = self.p["loss_fraction_rated"]["value"]
        self.curve = LookupCurve(
            self.p["load_fraction_breakpoints"]["value"],
            self.p["loss_fraction_breakpoints"]["value"],
        )

    def predict(self, inputs: dict) -> dict:
        """inputs: {load_fraction: 0..1, power_in?}. Returns loss fraction, efficiency."""
        lf = inputs.get("load_fraction", 1.0)
        loss = float(self.curve.lookup(lf))
        out = {"loss_fraction": loss, "efficiency": 1.0 - loss}
        p_in = inputs.get("power_in", None)
        if p_in is not None:
            out["power_loss"] = float(p_in) * loss
            out["power_out"] = float(p_in) * (1.0 - loss)
        return out

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"load_fraction": "fraction of rating (0-1)",
                       "power_in": "optional throughput power (unit of p_rated)"},
            "outputs": {"loss_fraction": "dimensionless", "efficiency": "dimensionless",
                        "power_out": "p_rated unit (if power_in)",
                        "power_loss": "p_rated unit (if power_in)"},
            "p_rated": self.p_rated, "loss_fraction_rated": self.loss_rated,
            "source": self.p["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print("get_info:", json.dumps(m.get_info(), indent=2))
    print("rated:", m.predict({"load_fraction": 1.0, "power_in": 100.0}))
    print("part-load 0.3:", m.predict({"load_fraction": 0.3, "power_in": 100.0}))
