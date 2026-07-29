"""EC060 — Solar Pond — F1b Incidence Angle Modifier — Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import SolarPondF1b


class ComponentModel:
    """Standardized interface for EC060 Solar Pond — F1b IAM model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SolarPondF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "irradiance_w_m2"    : float or array [W/m2]
                "incidence_angle_deg": float or array [deg] — solar zenith
                "T_lcz_degC"         : float or array [degC] — LCZ temperature
                "T_ambient_degC"     : float or array [degC]
            }
        Returns: useful_heat_w, efficiency, iam_factor, T_extraction_degC
        """
        return self._model.predict_all(
            inputs["irradiance_w_m2"],
            inputs["incidence_angle_deg"],
            inputs["T_lcz_degC"],
            inputs["T_ambient_degC"],
        )

    def get_info(self) -> dict:
        return {
            "name": "Solar Pond",
            "ec_id": "EC060",
            "fidelity": "F1b",
            "model": "Hottel-Whillier LCZ + Fresnel/Snell IAM for brine surface",
            "description": (
                "Q_u = A*[tau_pond*IAM(theta)*alpha_lcz*G - U_lcz*(T_lcz - T_amb)]; "
                "IAM from Fresnel reflectance + Snell's law refraction into brine"
            ),
            "inputs": {
                "irradiance_w_m2":     {"unit": "W/m2", "range": [0, 1100]},
                "incidence_angle_deg": {"unit": "deg",  "range": [0, 80]},
                "T_lcz_degC":          {"unit": "degC", "range": [30, 95]},
                "T_ambient_degC":      {"unit": "degC", "range": [0, 45]},
            },
            "outputs": {
                "useful_heat_w":     {"unit": "W"},
                "efficiency":        {"unit": "-"},
                "iam_factor":        {"unit": "-"},
                "T_extraction_degC": {"unit": "degC"},
            },
            "source": "Duffie & Beckman (2013) Ch.9; Singh et al. (2011)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC060 F1b — Solar Pond:")
    r = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 30.0,
                        "T_lcz_degC": 80.0, "T_ambient_degC": 25.0})
    for k, v in r.items():
        print(f"  {k}: {float(np.atleast_1d(v)[0]):.4f}")
