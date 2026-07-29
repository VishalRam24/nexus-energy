"""Standard predict interface for EC156 F0a COP-map model."""
import json
import os

from model import CopMap

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARAMS = os.path.join(_HERE, "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC156"
    component_name = "Geothermal Heat Pump (GHP)"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.p = json.load(f)
        mp = self.p["map"]
        self.model = CopMap(
            T_source_degC=mp["T_source_degC"],
            T_sink_degC=mp["T_sink_degC"],
            COP=mp["COP"],
            carnot_fraction=self.p["carnot_fraction"]["value"],
            aux_power=self.p["auxiliary_power"]["value"],
        )
        self.rated_capacity = self.p["rated_capacity"]["value"]

    def predict(self, inputs: dict) -> dict:
        """inputs: T_source [degC], T_sink [degC], optional Q_thermal_kW, part_load_ratio."""
        T_src = inputs["T_source"]
        T_sink = inputs["T_sink"]
        plr = inputs.get("part_load_ratio", 1.0)
        Q = inputs.get("Q_thermal_kW", self.rated_capacity)
        cop = self.model.cop(T_src, T_sink)
        p_el = self.model.electric_power_kW(Q, T_src, T_sink, plr)
        return {
            "COP": float(cop),
            "Q_thermal_kW": float(Q * plr),
            "electric_power_kW": float(p_el),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_source": "degC (ground loop)",
                "T_sink": "degC (supply)",
                "Q_thermal_kW": "kW (optional, default rated)",
                "part_load_ratio": "dimensionless (optional)",
            },
            "outputs": {
                "COP": "dimensionless",
                "Q_thermal_kW": "kW",
                "electric_power_kW": "kW",
            },
            "valid_ranges": self.p["valid_ranges"],
            "source": self.p["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    sample = {"T_source": 9.0, "T_sink": 45.0}
    print("get_info:", json.dumps(m.get_info(), indent=2))
    print("predict", sample, "->", m.predict(sample))
