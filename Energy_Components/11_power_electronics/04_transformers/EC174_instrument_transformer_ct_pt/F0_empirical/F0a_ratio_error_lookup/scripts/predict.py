"""F0a empirical predict interface for EC174 Instrument Transformer (CT/PT)."""
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
    component_id = "EC174"
    component_name = "Instrument Transformer (CT/PT)"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.p = json.load(f)
        self.ct_ratio = self.p["ct_ratio"]["value"]
        self.pt_ratio = self.p["pt_ratio"]["value"]
        self.max_err = self.p["max_ratio_error_pct"]["value"]
        self.curve = LookupCurve(
            self.p["load_fraction_breakpoints"]["value"],
            self.p["ratio_error_pct_breakpoints"]["value"],
        )

    def predict(self, inputs: dict) -> dict:
        """inputs: {load_fraction: 0..2, i_primary? (CT), v_primary? (PT)}."""
        lf = inputs.get("load_fraction", 1.0)
        err = float(self.curve.lookup(lf))
        out = {"ratio_error_pct": err, "within_class": bool(abs(err) <= self.max_err + 1e-9)}
        if "i_primary" in inputs:
            out["i_secondary"] = float(inputs["i_primary"]) / self.ct_ratio
        if "v_primary" in inputs:
            out["v_secondary"] = float(inputs["v_primary"]) * self.pt_ratio
        return out

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"load_fraction": "fraction of rated burden/current",
                       "i_primary": "optional CT primary current (A)",
                       "v_primary": "optional PT primary voltage (V)"},
            "outputs": {"ratio_error_pct": "%", "within_class": "bool",
                        "i_secondary": "A (if i_primary)", "v_secondary": "V (if v_primary)"},
            "ct_ratio": self.ct_ratio, "pt_ratio": self.pt_ratio,
            "source": self.p["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print("get_info:", json.dumps(m.get_info(), indent=2))
    print("rated:", m.predict({"load_fraction": 1.0, "i_primary": 1000.0, "v_primary": 11000.0}))
