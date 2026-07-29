"""F0a standardized prediction interface for EC168 MPPT Controller."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import EfficiencyCurveModel  # noqa: E402


class ComponentModel:
    component_id = "EC168"
    component_name = "MPPT Controller"
    fidelity = "F0a -- empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=None):
        if params_path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            params_path = os.path.join(here, "..", "data", "parameters.json")
        self._params_path = params_path
        self.model = EfficiencyCurveModel(params_path)

    def predict(self, inputs: dict) -> dict:
        """inputs: {'load_fraction': x} OR {'p_out': W}.
        returns efficiency, p_out, p_in, losses."""
        if "load_fraction" in inputs:
            lf = float(inputs["load_fraction"])
        elif "p_out" in inputs:
            lf = float(inputs["p_out"]) / self.model.p_rated
        else:
            raise KeyError("provide 'load_fraction' or 'p_out'")
        eta = float(self.model.efficiency_at(lf))
        p_out = lf * self.model.p_rated
        p_in = p_out / eta if eta > 0 else 0.0
        return {
            "efficiency": eta,
            "load_fraction": lf,
            "p_out": p_out,
            "p_in": p_in,
            "losses": p_in - p_out,
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"load_fraction": "0..1.x (dimensionless)",
                       "p_out": "W or VA (alternative input)"},
            "outputs": {"efficiency": "dimensionless", "p_out": "W/VA",
                        "p_in": "W/VA", "losses": "W"},
            "p_rated": self.model.p_rated,
            "rated_efficiency": self.model.rated_efficiency,
            "source": self.model.source,
        }


if __name__ == "__main__":
    cm = ComponentModel()
    print("Model:", cm.component_id, cm.component_name, "|", cm.fidelity)
    for lf in (0.1, 0.5, 1.0):
        print("  load=%.2f ->" % lf, cm.predict({"load_fraction": lf}))
