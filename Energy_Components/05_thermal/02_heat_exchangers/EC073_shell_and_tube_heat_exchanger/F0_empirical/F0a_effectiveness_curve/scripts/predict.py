"""F0a predict interface for Shell-and-Tube Heat Exchanger (EC073)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import EffectivenessCurve

_HERE = os.path.dirname(__file__)
_PARAMS = os.path.join(_HERE, "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC073"
    component_name = "Shell-and-Tube Heat Exchanger"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.p = json.load(f)
        lk = self.p["lookup"]
        self.curve = EffectivenessCurve(lk["NTU_breakpoints"]["value"],
                                        lk["effectiveness_table"]["value"])
        r = self.p["rated"]
        self.UA = r["UA"]["value"]
        self.Cmin = r["Cmin"]["value"]

    def predict(self, inputs):
        t_h = float(inputs.get("T_h_in", 80.0))
        t_c = float(inputs.get("T_c_in", 20.0))
        UA = float(inputs.get("UA", self.UA))
        Cmin = float(inputs.get("Cmin", self.Cmin))
        ntu = UA / Cmin if Cmin > 0 else 0.0
        eps = self.curve.effectiveness(ntu)
        q_max = Cmin * (t_h - t_c)
        q = eps * q_max
        return {"effectiveness": eps, "NTU": ntu, "Q_W": q, "Q_max_W": q_max}

    def get_info(self):
        return {"component_id": self.component_id, "component_name": self.component_name,
                "fidelity": self.fidelity, "version": self.version,
                "inputs": {"T_h_in": "degC", "T_c_in": "degC", "UA": "W/K", "Cmin": "W/K"},
                "outputs": {"effectiveness": "-", "NTU": "-", "Q_W": "W", "Q_max_W": "W"},
                "valid_ranges": self.p["valid_ranges"], "source": self.p["source"]}


if __name__ == "__main__":
    m = ComponentModel()
    print(m.component_id, m.component_name, "|", m.fidelity, "v" + m.version)
    print("sample predict:", m.predict({"T_h_in": 80.0, "T_c_in": 20.0}))
