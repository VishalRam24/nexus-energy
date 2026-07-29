"""EC002 — SOFC — F1a Polarization Curve — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import SOCFF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SOCFF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        j     = np.asarray(inputs["current_density"], dtype=float)
        T_c   = inputs.get("temperature", None)
        if T_c is not None:
            T_c = np.asarray(T_c, dtype=float)
        return self._model.predict(j, T_c)

    def get_info(self) -> dict:
        return {
            "name": "Solid Oxide Fuel Cell (SOFC)",
            "ec_id": "EC002",
            "fidelity": "F1a",
            "description": "V_cell = E_Nernst - V_act(BV) - V_ohm(ASR) - V_conc; H2/YSZ/air stack",
            "inputs": {
                "current_density": {"unit": "A/cm2", "range": [0.0, 1.8]},
                "temperature":     {"unit": "degC",  "range": [600.0, 1000.0], "default": 800.0},
            },
            "outputs": {
                "cell_voltage":    {"unit": "V"},
                "stack_voltage":   {"unit": "V"},
                "power_density":   {"unit": "W/cm2"},
                "stack_power_kw":  {"unit": "kW"},
                "efficiency":      {"unit": "dimensionless"},
            },
            "source": "Chan et al. (2001), J. Power Sources, 93, 130-140",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"current_density": 0.5, "temperature": 800.0})
    print(f"At j=0.5 A/cm2, T=800C: V_cell={float(r['cell_voltage']):.4f} V, "
          f"V_stack={float(r['stack_voltage']):.2f} V, "
          f"P_stack={float(r['stack_power_kw']):.3f} kW, "
          f"eta={float(r['efficiency']):.3f}")
