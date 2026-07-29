"""F0a empirical predict interface for EC126 Flywheel Energy Storage (FESS)."""
import json
import os

import numpy as np

try:
    from model import StorageRTECurve
except ImportError:  # when imported as a package
    from .model import StorageRTECurve

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARAMS = os.path.join(_HERE, "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC126"
    component_name = "Flywheel Energy Storage (FESS)"
    fidelity = "F0a — empirical lookup"
    version = "1.0.0"

    def __init__(self, params_path=_PARAMS):
        with open(params_path) as f:
            self.p = json.load(f)
        self.rte_rated = self.p["rte_rated"]["value"]
        self.curve = StorageRTECurve(
            frac_breakpoints=self.p["power_fraction_breakpoints"]["value"],
            rte_breakpoints=self.p["rte_breakpoints"]["value"],
            rte_rated=self.rte_rated,
            self_discharge_per_hr=self.p["self_discharge_per_hr"]["value"],
        )

    def predict(self, inputs: dict) -> dict:
        """inputs: {power_fraction: 0..1, energy_in_kwh?, idle_hours?}.

        Returns round-trip efficiency, deliverable energy out, and retained
        fraction after idle self-discharge.
        """
        pf = inputs.get("power_fraction", 1.0)
        rte = self.curve.round_trip_efficiency(pf)
        idle_hours = inputs.get("idle_hours", 0.0)
        retained = self.curve.retained_fraction(idle_hours)
        energy_in = inputs.get("energy_in_kwh", None)
        out = {
            "round_trip_efficiency": float(rte),
            "retained_fraction": float(retained),
        }
        if energy_in is not None:
            out["energy_out_kwh"] = float(energy_in) * float(rte) * float(retained)
        return out

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "power_fraction": "fraction of rated power (0-1)",
                "energy_in_kwh": "optional charge energy (kWh)",
                "idle_hours": "optional idle time for self-discharge (h)",
            },
            "outputs": {
                "round_trip_efficiency": "dimensionless",
                "retained_fraction": "dimensionless",
                "energy_out_kwh": "kWh (if energy_in_kwh given)",
            },
            "rte_rated": self.rte_rated,
            "source": self.p["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print("get_info:", json.dumps(m.get_info(), indent=2))
    print("rated:", m.predict({"power_fraction": 1.0, "energy_in_kwh": 1000.0}))
    print("part-load 0.25:", m.predict({"power_fraction": 0.25, "energy_in_kwh": 1000.0}))
