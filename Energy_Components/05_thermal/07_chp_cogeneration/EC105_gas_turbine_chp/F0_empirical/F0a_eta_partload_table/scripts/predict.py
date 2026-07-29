"""Standard predict interface for EC105 Gas Turbine CHP (with HRSG) (F0a CHP lookup)."""
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARAMS = os.path.join(_HERE, "..", "data", "parameters.json")

try:
    from model import EtaPartLoadTable
except ImportError:
    import sys
    sys.path.insert(0, _HERE)
    from model import EtaPartLoadTable


def _scalar(x):
    return float(np.asarray(x).item()) if np.ndim(x) == 0 else np.asarray(x)


class ComponentModel:
    component_id = "EC105"
    component_name = "Gas Turbine CHP (with HRSG)"
    fidelity = "F0a - empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.p = json.load(f)
        lk = self.p["lookup"]
        rt = self.p["rated"]
        self.tab = EtaPartLoadTable(
            lk["plr_breakpoints"]["value"],
            lk["eta_el_breakpoints"]["value"],
            lk["eta_th_breakpoints"]["value"],
            rt["P_el_rated"]["value"],
            rt["PLR_min"]["value"],
        )

    def predict(self, inputs: dict) -> dict:
        plr = inputs.get("part_load_ratio", inputs.get("plr", 1.0))
        eta_e = self.tab.eta_el(plr)
        eta_t = self.tab.eta_th(plr)
        p_el = self.tab.power_el(plr)
        return {
            "electrical_efficiency": _scalar(eta_e),
            "thermal_efficiency": _scalar(eta_t),
            "total_efficiency": _scalar(eta_e + eta_t),
            "power_electrical_W": _scalar(p_el),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"part_load_ratio": "dimensionless [0-1]"},
            "outputs": {
                "electrical_efficiency": "dimensionless",
                "thermal_efficiency": "dimensionless",
                "total_efficiency": "dimensionless",
                "power_electrical_W": "W",
            },
            "rated_power_el_W": self.p["rated"]["P_el_rated"]["value"],
            "eta_el_rated": self.p["rated"]["eta_el_rated"]["value"],
            "eta_th_rated": self.p["rated"]["eta_th_rated"]["value"],
            "source": self.p["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.component_id, m.component_name, "|", m.fidelity, "v" + m.version)
    for plr in (1.0, 0.6, 0.1):
        print(f"PLR={plr:.2f} ->", m.predict({"part_load_ratio": plr}))
