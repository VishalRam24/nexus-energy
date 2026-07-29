"""EC138 — Ocean Thermal Energy Conversion (OTEC) — F0a — standardized predict interface."""
import json, numpy as np
from pathlib import Path
from model import OtecF0a


class ComponentModel:
    component_id = "EC138"
    component_name = "Ocean Thermal Energy Conversion (OTEC)"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = OtecF0a(self.params)

    def predict(self, inputs: dict) -> dict:
        if "dT_c" in inputs:
            dT = np.asarray(inputs["dT_c"], dtype=float)
        else:
            dT = np.asarray(inputs["T_warm_c"], dtype=float) - np.asarray(inputs["T_cold_c"], dtype=float)
        return {
            "net_efficiency": self._model.net_efficiency(dT),
            "net_power_kw": self._model.net_power_kw(dT),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "dT_c": {"unit": "degC", "range": [10.0, 28.0], "note": "Warm-cold seawater temperature difference"},
                "T_warm_c": {"unit": "degC", "note": "Alternative: provide warm and cold temperatures"},
                "T_cold_c": {"unit": "degC"},
            },
            "outputs": {
                "net_efficiency": {"unit": "dimensionless"},
                "net_power_kw": {"unit": "kW"},
            },
            "source": self.params["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"dT_c": 21.0})
    print(f"At dT=21 C: eta_net={float(r['net_efficiency'])*100:.2f}%, P_net={float(r['net_power_kw']):.1f} kW")
