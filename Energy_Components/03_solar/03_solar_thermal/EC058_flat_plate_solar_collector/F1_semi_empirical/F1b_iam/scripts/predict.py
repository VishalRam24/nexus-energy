"""EC058 — Flat Plate Collector — F1b IAM — Standardized Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import FlatPlateCollectorF1b


class ComponentModel:
    """Standardized interface for EC058 Flat Plate Collector — F1b IAM model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = FlatPlateCollectorF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "irradiance_w_m2": W/m2,
                "incidence_angle_deg": degrees,
                "T_inlet_degC": degC,
                "T_ambient_degC": degC
            }
        """
        return self._model.predict_all(
            inputs["irradiance_w_m2"],
            inputs["incidence_angle_deg"],
            inputs["T_inlet_degC"],
            inputs["T_ambient_degC"],
        )

    def get_info(self) -> dict:
        return {
            "name": "Flat Plate Solar Collector",
            "ec_id": "EC058",
            "fidelity": "F1b",
            "description": "Hottel-Whillier model with Incidence Angle Modifier: IAM(theta) = 1 - b0*(1/cos(theta) - 1).",
            "inputs": {
                "irradiance_w_m2": {"unit": "W/m2", "range": [0.0, 1200.0]},
                "incidence_angle_deg": {"unit": "deg", "range": [0.0, 85.0]},
                "T_inlet_degC": {"unit": "degC", "range": [10.0, 90.0]},
                "T_ambient_degC": {"unit": "degC", "range": [-10.0, 45.0]},
            },
            "outputs": {
                "thermal_output_w": {"unit": "W"},
                "efficiency": {"unit": "dimensionless"},
                "iam_factor": {"unit": "dimensionless"},
                "T_outlet_degC": {"unit": "degC"},
            },
            "source": "Duffie & Beckman (2013), Ch.6; ASHRAE 93 (2010)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({
        "irradiance_w_m2": 800.0, "incidence_angle_deg": 30.0,
        "T_inlet_degC": 40.0, "T_ambient_degC": 20.0
    })
    print("\nAt G=800, theta=30, T_in=40, T_amb=20:")
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
