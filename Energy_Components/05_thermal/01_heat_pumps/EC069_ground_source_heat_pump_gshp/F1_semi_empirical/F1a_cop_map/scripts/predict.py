"""EC069 — GSHP — F1a COP Map — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import GSHPF1a


class ComponentModel:
    """Standardized interface for EC069 Ground-Source Heat Pump — F1a COP map model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = GSHPF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "T_source":        degC (0-20)  — ground loop supply temperature
                "T_sink":          degC (25-65) — heating distribution temperature
                "part_load_ratio": dimensionless (0-1), default=1.0
            }
        Returns:
            dict with cop [-], heating_capacity_kw [kW], electrical_input_kw [kW]
        """
        Ts  = np.asarray(inputs["T_source"], dtype=float)
        Tk  = np.asarray(inputs["T_sink"],   dtype=float)
        plr = np.asarray(inputs.get("part_load_ratio", 1.0), dtype=float)
        return {
            "cop":                  self._model.cop(Ts, Tk),
            "heating_capacity_kw":  self._model.heating_capacity(Ts, Tk, plr),
            "electrical_input_kw":  self._model.electrical_input(Ts, Tk, plr),
        }

    def get_info(self) -> dict:
        return {
            "name":        "Ground-Source Heat Pump (GSHP)",
            "ec_id":       "EC069",
            "fidelity":    "F1a",
            "description": (
                "COP = eta_Carnot * T_sink / (T_sink - T_source). "
                "Carnot fraction 0.50 (higher than ASHP 0.45) due to stable ground source. "
                "Rated at G10/W35: COP~4.5."
            ),
            "inputs": {
                "T_source":        {"unit": "degC", "range": [0.0, 20.0]},
                "T_sink":          {"unit": "degC", "range": [25.0, 65.0]},
                "part_load_ratio": {"unit": "-",    "range": [0.0, 1.0], "default": 1.0},
            },
            "outputs": {
                "cop":                 {"unit": "dimensionless"},
                "heating_capacity_kw": {"unit": "kW"},
                "electrical_input_kw": {"unit": "kW"},
            },
            "source": "Staffell et al. (2012); ASHRAE Handbook HVAC Applications (2019) Ch.34; EN 15450",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_source": 10.0, "T_sink": 35.0})
    print(f"At G10/W35: COP={float(r['cop']):.2f}, Q={float(r['heating_capacity_kw']):.1f}kW, W={float(r['electrical_input_kw']):.2f}kW")
