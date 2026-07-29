"""Standard ComponentModel interface for EC044 Monocrystalline Silicon PV (F0a)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import EfficiencyCurve  # noqa: E402

_HERE = os.path.dirname(__file__)
_PARAMS = os.path.join(_HERE, "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC044"
    component_name = "Monocrystalline Silicon PV"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.params = json.load(f)
        r = self.params["rated"]
        self.bifaciality = float(r.get('bifaciality_factor', {'value': 0.0})['value'])
        self.area = float(r["area"]["value"])
        self.p_stc = float(r["p_mp_stc"]["value"])
        self.curve = EfficiencyCurve(
            eta_stc=r["eta_stc"]["value"],
            gamma_pmp=r["gamma_pmp"]["value"],
            area=r["area"]["value"],
            irr_bp=self.params["irradiance_breakpoints"]["value"],
            rel_eff_bp=self.params["rel_eff_breakpoints"]["value"],
            bifaciality=self.bifaciality,
            stc_temp=r["stc_temperature"]["value"],
        )

    def predict(self, inputs: dict) -> dict:
        """inputs: irradiance [W/m2], cell_temperature [degC]."""
        g = float(inputs.get("irradiance", 1000.0))
        t = float(inputs.get("cell_temperature", 25.0))
        power = float(self.curve.power(g, t))
        return {
            "power": power,
            "efficiency": float(self.curve.efficiency(g, t)),
            "power_unit": "W",
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"irradiance": "W/m2", "cell_temperature": "degC"},
            "outputs": {"power": "W", "efficiency": "dimensionless"},
            "rated_power_stc_W": self.p_stc,
            "source": self.params["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print("INFO:", json.dumps(m.get_info(), indent=2))
    print("STC :", m.predict({"irradiance": 1000.0, "cell_temperature": 25.0}))
    print("Half:", m.predict({"irradiance": 500.0, "cell_temperature": 45.0}))
