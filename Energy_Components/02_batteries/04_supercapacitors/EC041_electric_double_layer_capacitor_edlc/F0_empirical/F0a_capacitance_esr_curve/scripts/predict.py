"""Standardized inference API for EDLC Supercapacitor (EC041) F0a capacitance/ESR lookup."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CapacitanceEsrCurve

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "parameters.json")


class ComponentModel:
    component_id = "EC041"
    component_name = "EDLC Supercapacitor"
    fidelity = "F0a -- empirical lookup (capacitance/ESR curve)"
    version = "1.0.0"

    def __init__(self, params_path=_DATA):
        with open(params_path) as f:
            self.params = json.load(f)
        self.curve = CapacitanceEsrCurve(self.params)

    def predict(self, inputs: dict) -> dict:
        """inputs: {'charge_C': Q} and/or {'current': I} -> voltage, energy, efficiency."""
        out = {}
        if "charge_C" in inputs:
            v = self.curve.voltage(inputs["charge_C"])
            out["voltage"] = float(v) if np.ndim(v) == 0 else v.tolist()
            e = self.curve.energy(v)
            out["energy_J"] = float(e) if np.ndim(e) == 0 else e.tolist()
        if "voltage" in inputs:
            e = self.curve.energy(inputs["voltage"])
            out["energy_J"] = float(e) if np.ndim(e) == 0 else e.tolist()
        cur = inputs.get("current", 0.0)
        eta = self.curve.roundtrip_efficiency(cur)
        out["roundtrip_efficiency"] = float(eta) if np.ndim(eta) == 0 else eta.tolist()
        return out

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {"charge_C": "stored charge (C)", "voltage": "terminal V",
                       "current": "|I| for efficiency (A)"},
            "outputs": {"voltage": "V", "energy_J": "J", "roundtrip_efficiency": "fraction"},
            "valid_ranges": {"voltage": [self.curve.V_min, self.curve.V_max]},
            "rated": {"capacitance_F": self.curve.C, "esr_Ohm": self.curve.esr,
                      "usable_energy_J": self.curve.usable_energy()},
            "source": self.params["source"],
        }


if __name__ == "__main__":
    m = ComponentModel()
    print("get_info:", json.dumps(m.get_info(), indent=2))
    Qmax = m.curve.C * m.curve.V_max
    for q in (0.0, 0.5 * Qmax, Qmax):
        print(f"  Q={q:>10.1f} C -> {m.predict({'charge_C': q, 'current': 10.0})}")
