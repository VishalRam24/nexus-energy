"""Standard ComponentModel interface for EC051 Dye-Sensitized Solar Cell (DSSC) (F0a)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import EfficiencyCurve  # noqa: E402

_HERE = os.path.dirname(__file__)
_PARAMS = os.path.join(_HERE, "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC051"
    component_name = "Dye-Sensitized Solar Cell (DSSC)"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.params = json.load(f)
        r = self.params["rated"]
        self.eta_stc = float(r["eta_stc"]["value"])
        self.irr_key = r["irradiance_key"]["value"]
        self.curve = EfficiencyCurve(
            eta_stc=r["eta_stc"]["value"],
            gamma=r["gamma_pmp"]["value"],
            irr_bp=self.params["irradiance_breakpoints"]["value"],
            rel_eff_bp=self.params["rel_eff_breakpoints"]["value"],
            stc_temp=r["stc_temperature"]["value"],
        )

    def predict(self, inputs: dict) -> dict:
        """inputs: G [W/m2] (or 'irradiance'), cell_temperature [degC]."""
        g = float(inputs.get(self.irr_key, inputs.get("irradiance", 1000.0)))
        t = float(inputs.get("cell_temperature", 25.0))
        return {
            "power_density": float(self.curve.power_density(g, t)),
            "efficiency": float(self.curve.efficiency(g, t)),
            "power_density_unit": "W/m2",
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {self.irr_key: "W/m2", "cell_temperature": "degC"},
            "outputs": {"power_density": "W/m2", "efficiency": "dimensionless"},
            "eta_stc": self.eta_stc,
            "source": self.params["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print("INFO:", json.dumps(m.get_info(), indent=2))
    print("STC :", m.predict({m.irr_key: 1000.0, "cell_temperature": 25.0}))
    print("Half:", m.predict({m.irr_key: 500.0, "cell_temperature": 45.0}))
