"""EC062 — HAWT Onshore — F1b Turbulence — Standardized Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import HAWTOnshoreF1b


class ComponentModel:
    """Standardized interface for EC062 HAWT Onshore — F1b turbulence correction."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = HAWTOnshoreF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "wind_speed_m_s": m/s,
                "turbulence_intensity": 0-0.30,
                "air_density": kg/m3 (optional, default 1.225)
            }
        """
        v = np.asarray(inputs["wind_speed_m_s"], dtype=float)
        ti = np.asarray(inputs.get("turbulence_intensity", 0.0), dtype=float)
        rho = np.asarray(inputs.get("air_density", 1.225), dtype=float)
        return {
            "power_kw": self._model.power(v, ti, rho),
            "power_coefficient": self._model.power_coefficient(v, ti, rho),
            "capacity_factor_correction": self._model.capacity_factor_correction(v, ti, rho),
        }

    def get_info(self) -> dict:
        return {
            "name": "HAWT Onshore Wind Turbine",
            "ec_id": "EC062",
            "fidelity": "F1b",
            "description": "Power curve with turbulence correction: P_corr = P(V) + 0.5*d2P/dV2*sigma_v^2.",
            "inputs": {
                "wind_speed_m_s": {"unit": "m/s", "range": [0.0, 30.0]},
                "turbulence_intensity": {"unit": "dimensionless", "range": [0.0, 0.30], "default": 0.0},
                "air_density": {"unit": "kg/m3", "range": [0.9, 1.4], "default": 1.225},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "power_coefficient": {"unit": "dimensionless"},
                "capacity_factor_correction": {"unit": "dimensionless"},
            },
            "source": "Albers et al. (2007), Wind Energy 10(4); IEC 61400-12-1",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"wind_speed_m_s": 10.0, "turbulence_intensity": 0.15})
    print(f"\nAt 10 m/s, TI=0.15:")
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
