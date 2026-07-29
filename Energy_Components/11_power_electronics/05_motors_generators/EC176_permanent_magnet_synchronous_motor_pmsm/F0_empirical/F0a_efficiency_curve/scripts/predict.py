"""F0a empirical predict interface for EC176 Permanent Magnet Synchronous Motor (PMSM)."""
import json
import os

import numpy as np

try:
    from model import EfficiencyCurve
except ImportError:  # imported as a package
    from .model import EfficiencyCurve

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARAMS = os.path.join(_HERE, "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC176"
    component_name = "Permanent Magnet Synchronous Motor (PMSM)"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.p = json.load(f)
        self.eta_rated = self.p["eta_rated"]["value"]
        self.curve = EfficiencyCurve(
            load_breakpoints=self.p["load_fraction_breakpoints"]["value"],
            eff_breakpoints=self.p["efficiency_breakpoints"]["value"],
            eta_rated=self.eta_rated,
        )
        self.p_rated = self.p["p_rated"]["value"]

    def predict(self, inputs: dict) -> dict:
        """inputs: {load_fraction: 0..1, power_in?}. Returns efficiency + losses."""
        lf = inputs.get("load_fraction", 1.0)
        eta = self.curve.efficiency(lf)
        loss_frac = self.curve.losses_fraction(lf)
        out = {"efficiency": float(eta), "loss_fraction": float(loss_frac)}
        p_in = inputs.get("power_in", None)
        if p_in is not None:
            out["power_out"] = float(p_in) * float(eta)
            out["power_loss"] = float(p_in) * (1.0 - float(eta))
        return out

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"load_fraction": "fraction of rated load (0-1)",
                       "power_in": "optional input power (same unit as p_rated)"},
            "outputs": {"efficiency": "dimensionless",
                        "loss_fraction": "dimensionless (loss/input)",
                        "power_out": "input unit (if power_in given)",
                        "power_loss": "input unit (if power_in given)"},
            "eta_rated": self.eta_rated,
            "p_rated": self.p_rated,
            "source": self.p["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print("get_info:", json.dumps(m.get_info(), indent=2))
    print("rated:", m.predict({"load_fraction": 1.0, "power_in": 1000.0}))
    print("part-load 0.25:", m.predict({"load_fraction": 0.25, "power_in": 1000.0}))
