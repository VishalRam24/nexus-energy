"""EC065 — Offshore Wind — F1b Turbulence — Standardized Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import OffshoreWindF1b


class ComponentModel:
    """Standardized interface for EC065 Offshore Wind — F1b turbulence + humid density."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = OffshoreWindF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "wind_speed_m_s": m/s,
                "turbulence_intensity": 0-0.30,
                "air_temperature_degC": degC (optional, default 15),
                "relative_humidity": 0-1 (optional, default 0.5)
            }
        """
        v = np.asarray(inputs["wind_speed_m_s"], dtype=float)
        ti = np.asarray(inputs.get("turbulence_intensity", 0.06), dtype=float)
        T = np.asarray(inputs.get("air_temperature_degC", 15.0), dtype=float)
        RH = np.asarray(inputs.get("relative_humidity", 0.5), dtype=float)

        rho = self._model.humid_air_density(T, RH)

        return {
            "power_kw": self._model.power(v, ti, T, RH),
            "power_coefficient": self._model.power_coefficient(v, ti, T, RH),
            "air_density_corrected": rho,
        }

    def get_info(self) -> dict:
        return {
            "name": "Offshore Fixed-Bottom Wind Turbine",
            "ec_id": "EC065",
            "fidelity": "F1b",
            "description": "Power curve with turbulence correction + humid air density: rho = P/(Rd*T)*(1 - 0.378*e_w/P).",
            "inputs": {
                "wind_speed_m_s": {"unit": "m/s", "range": [0.0, 30.0]},
                "turbulence_intensity": {"unit": "dimensionless", "range": [0.0, 0.30], "default": 0.06},
                "air_temperature_degC": {"unit": "degC", "range": [-10.0, 40.0], "default": 15.0},
                "relative_humidity": {"unit": "dimensionless", "range": [0.0, 1.0], "default": 0.5},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "power_coefficient": {"unit": "dimensionless"},
                "air_density_corrected": {"unit": "kg/m3"},
            },
            "source": "Albers et al. (2007); Buck (1981); IEC 61400-12-1",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"wind_speed_m_s": 10.0, "turbulence_intensity": 0.08,
                        "air_temperature_degC": 12.0, "relative_humidity": 0.80})
    print(f"\nAt 10 m/s, TI=0.08, T=12C, RH=80%:")
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
