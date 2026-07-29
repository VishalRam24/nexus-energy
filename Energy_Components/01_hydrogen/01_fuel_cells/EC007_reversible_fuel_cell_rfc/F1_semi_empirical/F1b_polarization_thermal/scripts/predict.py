"""
EC007 -- Reversible Fuel Cell (RFC) -- F1b Polarization-Thermal -- Standardized Predict Interface

Usage:
    model = ComponentModel()
    result = model.predict({"current_density": 0.5, "temperature": 353.15, "mode": "fc"})
"""

import json
import numpy as np
from pathlib import Path
from model import RFCThermalModel


class ComponentModel:
    """Standardized interface for EC007 RFC -- F1b polarization-thermal model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = RFCThermalModel(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict RFC operating point.

        Args:
            inputs: dict with keys:
                - current_density (A/cm2): Operating current density (>= 0)
                - temperature (K):         Stack temperature
                - mode (str):              'fc' or 'electrolyser' (default: 'fc')

        Returns:
            dict with operating-point outputs
        """
        j    = np.asarray(inputs["current_density"], dtype=float)
        T    = np.asarray(inputs.get("temperature", self._model.T_ref), dtype=float)
        mode = inputs.get("mode", "fc")

        out = self._model.evaluate(j, T, mode=mode)

        return {
            "cell_voltage_V":              out["cell_voltage"],
            "power_stack_kW":              out["power_stack_kW"],
            "efficiency":                  out["efficiency"],
            "heat_area_W_cm2":             out["heat_area"],
            "heat_stack_W":                out["heat_stack_W"],
            "membrane_resistance_ohm_cm2": out["membrane_resistance"],
            "E_nernst_V":                  out["E_nernst"],
            "V_act_V":                     out["V_act"],
            "V_ohm_V":                     out["V_ohm"],
            "V_conc_V":                    out["V_conc"],
            "dTdt_K_s":                    out["dTdt_K_s"],
        }

    def get_info(self) -> dict:
        return {
            "name": "Reversible Fuel Cell (RFC)",
            "ec_id": "EC007",
            "fidelity": "F1b",
            "description": (
                "Temperature-dependent polarization curve for RFC in FC and electrolyser modes. "
                "Arrhenius i0(T), Nafion sigma(T), lumped thermal balance."
            ),
            "inputs": {
                "current_density": {"unit": "A/cm2",  "range": [0.0, 2.5]},
                "temperature":     {"unit": "K",       "range": [313.15, 363.15]},
                "mode":            {"values": ["fc", "electrolyser"]},
            },
            "outputs": {
                "cell_voltage_V":              {"unit": "V"},
                "power_stack_kW":              {"unit": "kW"},
                "efficiency":                  {"unit": "dimensionless"},
                "heat_area_W_cm2":             {"unit": "W/cm2"},
                "heat_stack_W":                {"unit": "W"},
                "membrane_resistance_ohm_cm2": {"unit": "ohm cm2"},
                "E_nernst_V":                  {"unit": "V"},
                "V_act_V":                     {"unit": "V"},
                "V_ohm_V":                     {"unit": "V"},
                "V_conc_V":                    {"unit": "V"},
                "dTdt_K_s":                    {"unit": "K/s"},
            },
            "source": "Amphlett (1995); Springer (1991); Grigoriev (2020); Ito (2012)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    print("\n-- FC Mode --")
    for j in [0.1, 0.5, 1.0, 1.5]:
        r = model.predict({"current_density": j, "temperature": 353.15, "mode": "fc"})
        print(f"  j={j:.1f}: V={float(r['cell_voltage_V']):.3f} V, "
              f"eta={float(r['efficiency']):.3f}, "
              f"Q={float(r['heat_area_W_cm2']):.3f} W/cm2")
    print("\n-- Electrolyser Mode --")
    for j in [0.1, 0.5, 1.0, 2.0]:
        r = model.predict({"current_density": j, "temperature": 353.15, "mode": "electrolyser"})
        print(f"  j={j:.1f}: V={float(r['cell_voltage_V']):.3f} V, "
              f"eta={float(r['efficiency']):.3f}")
