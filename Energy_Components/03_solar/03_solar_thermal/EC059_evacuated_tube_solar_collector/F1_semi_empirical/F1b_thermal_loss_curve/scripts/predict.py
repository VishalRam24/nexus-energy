"""EC059 — Evacuated Tube Solar Collector — F1b Thermal Loss Curve — Standardized Interface"""

import json
import numpy as np
from pathlib import Path
from model import EvacuatedTubeF1b


class ComponentModel:
    """Standardized interface for EC059 ETC — F1b second-order U_L(T) + IAM model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = EvacuatedTubeF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "irradiance_w_m2": W/m2 (0-1200),
                "T_inlet_degC": degC (10-180),
                "T_ambient_degC": degC (-20 to 45),
                "incidence_angle_deg": degrees (0-80) [optional, default 0]
            }
        Returns:
            useful_heat_w, efficiency, T_outlet_c, T_mean_c,
            delta_T_m, U_L_eff_w_m2k, iam
        """
        G = np.asarray(inputs["irradiance_w_m2"], dtype=float)
        T_in = np.asarray(inputs["T_inlet_degC"], dtype=float)
        T_amb = np.asarray(inputs["T_ambient_degC"], dtype=float)
        theta = float(inputs.get("incidence_angle_deg", 0.0))
        return self._model.predict_all(G, T_in, T_amb, theta)

    def get_info(self) -> dict:
        return {
            "name": "Evacuated Tube Solar Collector",
            "ec_id": "EC059",
            "fidelity": "F1b",
            "description": (
                "ISO 9806 second-order efficiency equation with U_L(DeltaT) = a1 + a2*DeltaT "
                "and incidence angle modifier IAM(theta). F1b extends F1a (constant U_L) with: "
                "(1) weak temperature-dependent U_L (a2=0.012 W/m2K2 for vacuum insulation, "
                "much lower than flat plate a2~0.06), and (2) IAM for off-normal incidence. "
                "Iterative solution for mean fluid temperature."
            ),
            "inputs": {
                "irradiance_w_m2":     {"unit": "W/m2",  "range": [0.0, 1200.0]},
                "T_inlet_degC":        {"unit": "degC",  "range": [10.0, 180.0]},
                "T_ambient_degC":      {"unit": "degC",  "range": [-20.0, 45.0]},
                "incidence_angle_deg": {"unit": "deg",   "range": [0.0, 80.0], "default": 0.0},
            },
            "outputs": {
                "useful_heat_w":    {"unit": "W"},
                "efficiency":       {"unit": "dimensionless"},
                "T_outlet_c":       {"unit": "degC"},
                "T_mean_c":         {"unit": "degC"},
                "delta_T_m":        {"unit": "K", "note": "T_mean - T_amb"},
                "U_L_eff_w_m2k":    {"unit": "W/m2K", "note": "Effective U_L at operating point"},
                "iam":              {"unit": "dimensionless"},
            },
            "source": "Duffie & Beckman (2013) Ch.6; ISO 9806:2017; SRCC OG-100",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"irradiance_w_m2": 800.0, "T_inlet_degC": 60.0, "T_ambient_degC": 20.0})
    print(f"\nAt G=800, T_in=60C, T_amb=20C, theta=0:")
    for k, v in r.items():
        print(f"  {k}: {float(np.asarray(v)):.4f}")
