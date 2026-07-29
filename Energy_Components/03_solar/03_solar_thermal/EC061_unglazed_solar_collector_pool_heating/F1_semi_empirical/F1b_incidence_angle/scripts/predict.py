"""EC061 — Unglazed Solar Collector (Pool Heating) — F1b IAM — Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import UnglazedCollectorF1b


class ComponentModel:
    """Standardized interface for EC061 Unglazed Solar Collector — F1b model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = UnglazedCollectorF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "irradiance_w_m2"    : float or array [W/m2]
                "incidence_angle_deg": float or array [deg]
                "T_inlet_degC"       : float or array [degC] — pool water temperature
                "T_ambient_degC"     : float or array [degC]
                "v_wind_m_s"         : float or array [m/s] (optional, default 2.0)
            }
        Returns: useful_heat_w, efficiency, iam_factor, T_outlet_degC,
                 U_L_effective, Q_sky_loss_w
        """
        return self._model.predict_all(
            inputs["irradiance_w_m2"],
            inputs["incidence_angle_deg"],
            inputs["T_inlet_degC"],
            inputs["T_ambient_degC"],
            inputs.get("v_wind_m_s", 2.0),
        )

    def get_info(self) -> dict:
        return {
            "name": "Unglazed Solar Collector (Pool Heating)",
            "ec_id": "EC061",
            "fidelity": "F1b",
            "model": "HWB + ASHRAE b0 IAM + wind-corrected U_L + sky radiation",
            "description": (
                "Q_u = A*F_R*[IAM*tau_alpha*G - U_L(v_wind)*(T_in - T_amb)] - F_R*Q_sky; "
                "U_L(v) = U_L0 + U_wind*v_wind; IAM = 1 - b0*(1/cos(theta)-1)"
            ),
            "inputs": {
                "irradiance_w_m2":     {"unit": "W/m2", "range": [0, 1200]},
                "incidence_angle_deg": {"unit": "deg",  "range": [0, 85]},
                "T_inlet_degC":        {"unit": "degC", "range": [15, 35]},
                "T_ambient_degC":      {"unit": "degC", "range": [5, 40]},
                "v_wind_m_s":          {"unit": "m/s",  "range": [0, 10], "default": 2.0},
            },
            "outputs": {
                "useful_heat_w":  {"unit": "W"},
                "efficiency":     {"unit": "-"},
                "iam_factor":     {"unit": "-"},
                "T_outlet_degC":  {"unit": "degC"},
                "U_L_effective":  {"unit": "W/m2K"},
                "Q_sky_loss_w":   {"unit": "W"},
            },
            "source": "Duffie & Beckman (2013) Ch.6,10; ASHRAE 93 (2010); ISO 9806:2017",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC061 F1b — Unglazed Solar Collector (Pool Heating):")
    r = model.predict({"irradiance_w_m2": 800.0, "incidence_angle_deg": 30.0,
                        "T_inlet_degC": 26.0, "T_ambient_degC": 20.0, "v_wind_m_s": 3.0})
    for k, v in r.items():
        print(f"  {k}: {float(np.atleast_1d(v)[0]):.4f}")
