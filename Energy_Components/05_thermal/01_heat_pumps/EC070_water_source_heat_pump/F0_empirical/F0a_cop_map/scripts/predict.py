"""F0a predict interface for Water-Source Heat Pump (EC070)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import CopMap

_HERE = os.path.dirname(__file__)
_PARAMS = os.path.join(_HERE, "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC070"
    component_name = "Water-Source Heat Pump"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.p = json.load(f)
        lk = self.p["lookup"]
        self.map = CopMap(lk["T_source_breakpoints"]["value"],
                          lk["T_sink_breakpoints"]["value"],
                          lk["COP_table"]["value"])
        self.aux = self.p["rated"]["auxiliary_power"]["value"]
        self.cap = self.p["rated"]["rated_capacity"]["value"]

    def predict(self, inputs):
        t_src = float(inputs.get("T_source", self.p["rated"]["rated_T_source"]["value"]))
        t_sink = float(inputs.get("T_sink", self.p["rated"]["rated_T_sink"]["value"]))
        plr = float(inputs.get("part_load_ratio", 1.0))
        cop = self.map.cop(t_src, t_sink)
        q_th = self.cap * plr
        p_elec = q_th / cop + self.aux if cop > 0 else float("inf")
        return {"COP": cop, "Q_thermal_kW": q_th,
                "P_input_kW": p_elec, "part_load_ratio": plr}

    def get_info(self):
        return {"component_id": self.component_id, "component_name": self.component_name,
                "fidelity": self.fidelity, "version": self.version,
                "inputs": {"T_source": "degC", "T_sink": "degC", "part_load_ratio": "-"},
                "outputs": {"COP": "-", "Q_thermal_kW": "kW", "P_input_kW": "kW"},
                "valid_ranges": self.p["valid_ranges"], "source": self.p["source"]}


if __name__ == "__main__":
    m = ComponentModel()
    print(m.component_id, m.component_name, "|", m.fidelity, "v" + m.version)
    print("sample predict:", m.predict({}))
