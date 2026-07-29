"""F0a predict interface for Molten Salt TES (EC079)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import CapacityLookup

_HERE = os.path.dirname(__file__)
_PARAMS = os.path.join(_HERE, "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC079"
    component_name = "Molten Salt TES"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.p = json.load(f)
        lk = self.p["lookup"]
        self.lut = CapacityLookup(lk["SOC_breakpoints"]["value"],
                                  lk["usable_energy_table"]["value"])
        r = self.p["rated"]
        self.E_cap = r["E_capacity"]["value"]
        self.eta_rt = r["round_trip_efficiency"]["value"]

    def predict(self, inputs):
        soc = float(inputs.get("soc", 0.5))
        e_avail = self.lut.energy_at(soc)
        e_deliverable = e_avail * self.eta_rt
        return {"soc": soc, "energy_stored_kWh": e_avail,
                "energy_deliverable_kWh": e_deliverable,
                "round_trip_efficiency": self.eta_rt}

    def get_info(self):
        return {"component_id": self.component_id, "component_name": self.component_name,
                "fidelity": self.fidelity, "version": self.version,
                "inputs": {"soc": "-"},
                "outputs": {"energy_stored_kWh": "kWh", "energy_deliverable_kWh": "kWh",
                            "round_trip_efficiency": "-"},
                "valid_ranges": self.p["valid_ranges"], "source": self.p["source"]}


if __name__ == "__main__":
    m = ComponentModel()
    print(m.component_id, m.component_name, "|", m.fidelity, "v" + m.version)
    print("sample predict:", m.predict({"soc": 0.5}))
