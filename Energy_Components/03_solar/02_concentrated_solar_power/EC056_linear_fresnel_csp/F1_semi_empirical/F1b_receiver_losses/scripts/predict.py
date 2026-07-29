"""EC056 — Linear Fresnel CSP — F1b Receiver Losses — Standardized Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import LinearFresnelF1b


class ComponentModel:
    """Standardized interface for EC056 Linear Fresnel CSP — F1b receiver loss model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = LinearFresnelF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "dni_w_m2": W/m2 (0-1200),
                "T_htf_in_degC": degC (150-400),
                "T_htf_out_degC": degC (150-400),
                "T_ambient_degC": degC,
                "theta_L_deg": longitudinal incidence angle (deg, 0-80),
                "theta_T_deg": transversal incidence angle (deg, 0-60)
            }
        Returns:
            thermal_output_kw_per_m, optical_efficiency, thermal_efficiency,
            receiver_loss_kw_per_m, iam_longitudinal, iam_transversal
        """
        return self._model.predict_all(
            inputs["dni_w_m2"],
            inputs["T_htf_in_degC"],
            inputs["T_htf_out_degC"],
            inputs["T_ambient_degC"],
            inputs["theta_L_deg"],
            inputs.get("theta_T_deg", 0.0),
        )

    def get_info(self) -> dict:
        return {
            "name": "Linear Fresnel CSP",
            "ec_id": "EC056",
            "fidelity": "F1b",
            "description": (
                "Linear Fresnel collector with two-axis IAM (longitudinal: cos(theta_L); "
                "transversal: 1 - b_T*theta_T^2), end-loss, and physics-based receiver "
                "heat loss (convection + radiation). Cavity receiver geometry."
            ),
            "inputs": {
                "dni_w_m2": {"unit": "W/m2", "range": [0.0, 1200.0]},
                "T_htf_in_degC": {"unit": "degC"},
                "T_htf_out_degC": {"unit": "degC"},
                "T_ambient_degC": {"unit": "degC"},
                "theta_L_deg": {"unit": "deg", "range": [0.0, 80.0]},
                "theta_T_deg": {"unit": "deg", "range": [0.0, 60.0], "optional": True},
            },
            "outputs": {
                "thermal_output_kw_per_m": {"unit": "kW/m"},
                "optical_efficiency": {"unit": "dimensionless"},
                "thermal_efficiency": {"unit": "dimensionless"},
                "receiver_loss_kw_per_m": {"unit": "kW/m"},
                "iam_longitudinal": {"unit": "dimensionless"},
                "iam_transversal": {"unit": "dimensionless"},
            },
            "source": "Zhu et al. (2014) Solar Energy; Häberle et al. (2002)",
            "library": "NumPy",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({
        "dni_w_m2": 800.0, "T_htf_in_degC": 240.0, "T_htf_out_degC": 300.0,
        "T_ambient_degC": 25.0, "theta_L_deg": 15.0, "theta_T_deg": 5.0
    })
    print("\nAt DNI=800, T_htf=240-300C, theta_L=15, theta_T=5:")
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
