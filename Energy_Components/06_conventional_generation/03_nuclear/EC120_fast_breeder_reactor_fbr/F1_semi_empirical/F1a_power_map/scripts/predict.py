"""EC120 -- Fast Breeder Reactor (FBR) -- F1a Power-Map -- Standardized Predict Interface"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import FBRF1a


class ComponentModel:
    component_id = "EC120"

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = FBRF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            load_factor : float  -- fraction of rated thermal power [0.25 – 1.0]
        returns:
            load_factor_clamped, P_thermal_mw, P_electric_mw, eta_thermal
        """
        lf = float(inputs.get("load_factor", 1.0))
        return self._model.predict(lf)

    def get_info(self) -> dict:
        m = self._model
        return {
            "name":        "Fast Breeder Reactor (FBR, Na-cooled)",
            "ec_id":       "EC120",
            "fidelity":    "F1a",
            "model":       "Power Map (P_elec = eta * P_thermal * load_factor)",
            "description": (
                f"Sodium-cooled FBR power map. P_thermal={m.P_thermal:.0f} MW_th, "
                f"eta={m.eta:.2f}, T_out={m.T_out:.0f} degC. "
                f"Load range [{m.load_min:.2f}, {m.load_max:.1f}]."
            ),
            "inputs":  {"load_factor": {"unit": "dimensionless", "range": [0.25, 1.0]}},
            "outputs": {
                "load_factor_clamped": {"unit": "dimensionless"},
                "P_thermal_mw":        {"unit": "MW_th"},
                "P_electric_mw":       {"unit": "MW_e"},
                "eta_thermal":         {"unit": "dimensionless"},
            },
            "source": "Guidez & Prele (2017); IAEA TECDOC-1689 (2012)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")
    for lf in [0.25, 0.5, 0.75, 1.0]:
        r = model.predict({"load_factor": lf})
        print(f"  LF={lf:.2f}: P_th={r['P_thermal_mw']:.0f} MW_th, P_el={r['P_electric_mw']:.0f} MW_e")
