"""F0a empirical predict interface for EC188 Superconducting Magnetic Energy Storage (SMES)."""
import json
import os

import numpy as np

try:
    from model import LookupCurve
except ImportError:
    from .model import LookupCurve

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARAMS = os.path.join(_HERE, "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC188"
    component_name = "Superconducting Magnetic Energy Storage (SMES)"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.p = json.load(f)
        self.L = self.p["L_H"]["value"]
        self.I_max = self.p["I_max_A"]["value"]
        self.rte_rated = self.p["rte_rated"]["value"]
        self.p_cryo = self.p["P_cryo_MW"]["value"]
        self.curve = LookupCurve(
            self.p["power_fraction_breakpoints"]["value"],
            self.p["rte_breakpoints"]["value"],
        )

    def stored_energy_mj(self, current_a):
        """E = 0.5 * L * I^2 in MJ."""
        return 0.5 * self.L * float(current_a) ** 2 / 1e6

    def predict(self, inputs: dict) -> dict:
        """inputs: {power_fraction: 0..1, energy_in_mwh?, current_a?, idle_hours?}."""
        pf = inputs.get("power_fraction", 1.0)
        rte = float(self.curve.lookup(pf))
        out = {"round_trip_efficiency": rte}
        if "current_a" in inputs:
            out["stored_energy_mj"] = self.stored_energy_mj(inputs["current_a"])
        e_in = inputs.get("energy_in_mwh", None)
        if e_in is not None:
            idle = inputs.get("idle_hours", 0.0)
            cryo_loss = self.p_cryo * float(idle) / 1.0  # MWh drawn over idle hours
            out["energy_out_mwh"] = max(0.0, float(e_in) * rte - cryo_loss)
        return out

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"power_fraction": "fraction of rated power",
                       "current_a": "optional coil current (A)",
                       "energy_in_mwh": "optional charge energy (MWh)",
                       "idle_hours": "optional idle time for cryo loss (h)"},
            "outputs": {"round_trip_efficiency": "dimensionless",
                        "stored_energy_mj": "MJ (if current_a)",
                        "energy_out_mwh": "MWh (if energy_in_mwh)"},
            "rte_rated": self.rte_rated, "L_H": self.L, "I_max_A": self.I_max,
            "source": self.p["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print("get_info:", json.dumps(m.get_info(), indent=2))
    print("at I_max:", m.predict({"power_fraction": 1.0, "current_a": m.I_max}))
