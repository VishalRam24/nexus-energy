"""
EC041 — EDLC Supercapacitor — F1a RC Model — Standardized Predict Interface

Inputs: capacitor voltage v_cap and terminal current.
Outputs: terminal voltage, SOC, stored energy, power, dV/dt.
"""

import json
import numpy as np
from pathlib import Path
from model import EDLCF1a


class ComponentModel:
    """Standardized interface for EC041 EDLC — F1a linear RC model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = EDLCF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "v_cap":   capacitor (state) voltage in V,
                "current": terminal current in A (positive=discharge)
            }
        Returns:
            {
                "voltage":      terminal voltage in V (= v_cap - I*ESR),
                "soc":          v_cap / v_max in [0, 1],
                "charge":       stored charge in C,
                "stored_energy": stored energy in J,
                "power":        terminal power in W,
                "dvcap_dt":     capacitor voltage rate in V/s,
            }
        """
        v_cap   = np.asarray(inputs["v_cap"],   dtype=float)
        current = np.asarray(inputs["current"], dtype=float)

        return {
            "voltage":       self._model.terminal_voltage(v_cap, current),
            "soc":           self._model.soc(v_cap),
            "charge":        self._model.charge(v_cap),
            "stored_energy": self._model.stored_energy(v_cap),
            "power":         self._model.power(v_cap, current),
            "dvcap_dt":      self._model.vcap_derivative(v_cap, current),
        }

    def get_info(self) -> dict:
        return {
            "name": "Electric Double-Layer Capacitor (EDLC) Supercapacitor",
            "ec_id": "EC041",
            "fidelity": "F1a",
            "description": "Linear RC model: V_term = V_cap - I*ESR; dV_cap/dt = -(I + V_cap/R_leak)/C",
            "inputs": {
                "v_cap":   {"unit": "V", "range": [0.0, 2.7]},
                "current": {"unit": "A", "range": [-400.0, 400.0],
                             "note": "positive=discharge, negative=charge"},
            },
            "outputs": {
                "voltage":       {"unit": "V"},
                "soc":           {"unit": "dimensionless"},
                "charge":        {"unit": "C"},
                "stored_energy": {"unit": "J"},
                "power":         {"unit": "W"},
                "dvcap_dt":      {"unit": "V/s"},
            },
            "source": "Maxwell BCAP3000 datasheet; Conway (1999), Electrochemical Supercapacitors",
            "license": "BSD-3 (model equations)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"v_cap": 2.5, "current": 100.0})
    print(f"\nAt V_cap=2.5V, I=100A discharge:")
    print(f"  V_term:        {float(r['voltage']):.4f} V")
    print(f"  SOC:           {float(r['soc']):.4f}")
    print(f"  Stored energy: {float(r['stored_energy']):.1f} J")
    print(f"  Power:         {float(r['power']):.1f} W")
    print(f"  dV_cap/dt:     {float(r['dvcap_dt']):.4f} V/s")
