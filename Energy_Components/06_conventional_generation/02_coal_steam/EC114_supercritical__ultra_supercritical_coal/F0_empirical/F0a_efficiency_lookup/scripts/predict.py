"""Standardized prediction interface for the EC114 F0a efficiency lookup."""
import json
import os
import numpy as np

try:
    from model import EfficiencyLookup
except ImportError:
    from .model import EfficiencyLookup

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARAMS = os.path.join(_HERE, "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC114"
    component_name = "Supercritical / Ultra-Supercritical Coal Plant"
    fidelity = "F0a - empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.params = json.load(f)
        u = self.params["unit"]
        self.P_rated = u["P_rated"]["value"]
        self.lookup = EfficiencyLookup(
            plr_bp=u["plr_breakpoints"]["value"],
            eta_bp=u["eta_breakpoints"]["value"],
            eta_rated=u["eta_rated"]["value"],
            T_amb_ref=u["T_amb_ref"]["value"],
            f_amb_coeff=u["f_amb_coeff"]["value"],
            PLR_min=u["PLR_min"]["value"],
        )

    def predict(self, inputs: dict) -> dict:
        plr = inputs.get("part_load_ratio", 1.0)
        T_amb = inputs.get("ambient_temp_c", None)
        eta = self.lookup.efficiency(plr, T_amb)
        P_out = self.P_rated * np.asarray(plr, dtype=float)
        fuel_thermal = np.where(eta > 0, P_out / eta, 0.0)
        return {
            "efficiency": eta,
            "power_output_mw": P_out,
            "fuel_power_mw": fuel_thermal,
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "part_load_ratio": "dimensionless (0.3-1.0)",
                "ambient_temp_c": "degC (optional)",
            },
            "outputs": {
                "efficiency": "dimensionless net LHV",
                "power_output_mw": "MW_e",
                "fuel_power_mw": "MW_e (LHV)",
            },
            "rated_power": self.P_rated,
            "source": self.params["unit"]["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    print("sample @ PLR=1.0:", m.predict({"part_load_ratio": 1.0, "ambient_temp_c": 15.0}))
    print("sample @ PLR=0.5:", m.predict({"part_load_ratio": 0.5, "ambient_temp_c": 30.0}))
