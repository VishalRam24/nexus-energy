"""EC054 — Parabolic Trough CSP — F1b Receiver Losses — Standardized Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import ParabolicTroughF1b


class ComponentModel:
    """Standardized interface for EC054 Parabolic Trough CSP — F1b receiver losses."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ParabolicTroughF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "DNI_w_m2": W/m2,
                "T_htf_in_degC": degC,
                "T_htf_out_degC": degC,
                "T_ambient_degC": degC,
                "incidence_angle_deg": deg
            }
        """
        return self._model.predict_all(
            inputs["DNI_w_m2"],
            inputs["T_htf_in_degC"],
            inputs["T_htf_out_degC"],
            inputs["T_ambient_degC"],
            inputs["incidence_angle_deg"],
        )

    def get_info(self) -> dict:
        return {
            "name": "Parabolic Trough CSP",
            "ec_id": "EC054",
            "fidelity": "F1b",
            "description": "Detailed receiver loss model: convective + radiative losses, polynomial IAM, end loss factor.",
            "inputs": {
                "DNI_w_m2": {"unit": "W/m2", "range": [0.0, 1000.0]},
                "T_htf_in_degC": {"unit": "degC", "range": [100.0, 400.0]},
                "T_htf_out_degC": {"unit": "degC", "range": [100.0, 400.0]},
                "T_ambient_degC": {"unit": "degC", "range": [0.0, 50.0]},
                "incidence_angle_deg": {"unit": "deg", "range": [0.0, 80.0]},
            },
            "outputs": {
                "thermal_output_kw_per_m": {"unit": "kW/m"},
                "optical_efficiency": {"unit": "dimensionless"},
                "thermal_efficiency": {"unit": "dimensionless"},
                "receiver_loss_kw_per_m": {"unit": "kW/m"},
            },
            "source": "Forristall (2003), NREL/TP-550-34169; Dudley et al. (1994), SAND94-1884",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({
        "DNI_w_m2": 800.0, "T_htf_in_degC": 290.0, "T_htf_out_degC": 390.0,
        "T_ambient_degC": 25.0, "incidence_angle_deg": 10.0
    })
    print("\nAt DNI=800, T_htf=290/390, T_amb=25, theta=10:")
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
