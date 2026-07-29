"""EC066 — Offshore Floating Wind — F1b Turbulence + Pitch — Standardized Predict Interface"""

import json
import numpy as np
from pathlib import Path
from model import FloatingOffshoreWindF1b


class ComponentModel:
    """Standardized interface for EC066 Floating Offshore Wind — F1b."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = FloatingOffshoreWindF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "wind_speed_m_s": m/s,
                "turbulence_intensity": 0–0.20 (optional, default 0.07),
                "platform_pitch_deg": degrees (optional, default 0.0),
                "air_temperature_degC": degC (optional, default 12.0),
                "relative_humidity": 0–1 (optional, default 0.80)
            }
        Returns:
            {
                "power_kw": kW,
                "power_coefficient": dimensionless,
                "air_density": kg/m3,
                "pitch_factor": dimensionless (cos^2 theta)
            }
        """
        v = np.asarray(inputs["wind_speed_m_s"], dtype=float)
        ti = np.asarray(inputs.get("turbulence_intensity", 0.07), dtype=float)
        pitch = np.asarray(inputs.get("platform_pitch_deg", 0.0), dtype=float)
        T = np.asarray(inputs.get("air_temperature_degC", 12.0), dtype=float)
        RH = np.asarray(inputs.get("relative_humidity", 0.80), dtype=float)

        rho = self._model.humid_air_density(T, RH)
        pitch_factor = self._model.platform_pitch_factor(pitch)

        return {
            "power_kw": self._model.power(v, ti, pitch, T, RH),
            "power_coefficient": self._model.power_coefficient(v, ti, pitch, T, RH),
            "air_density": rho,
            "pitch_factor": pitch_factor,
        }

    def get_info(self) -> dict:
        return {
            "name": "Offshore Floating Wind Turbine",
            "ec_id": "EC066",
            "fidelity": "F1b",
            "description": (
                "Power curve + marine humid air density (Buck eq.) "
                "+ wave-induced pitch penalty cos^2(theta) "
                "+ offshore TI correction (lower TI 0.06-0.10)."
            ),
            "inputs": {
                "wind_speed_m_s": {"unit": "m/s", "range": [0.0, 30.0]},
                "turbulence_intensity": {"unit": "dimensionless", "range": [0.0, 0.20], "default": 0.07},
                "platform_pitch_deg": {"unit": "deg", "range": [0.0, 15.0], "default": 0.0},
                "air_temperature_degC": {"unit": "degC", "range": [-10.0, 40.0], "default": 12.0},
                "relative_humidity": {"unit": "dimensionless", "range": [0.0, 1.0], "default": 0.80},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "power_coefficient": {"unit": "dimensionless"},
                "air_density": {"unit": "kg/m3"},
                "pitch_factor": {"unit": "dimensionless"},
            },
            "source": "Gaertner et al. (2020); Allen et al. (2020); Buck (1981); Jiang et al. (2014)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    for pitch in [0, 3, 6, 10]:
        r = model.predict({"wind_speed_m_s": 10.0, "turbulence_intensity": 0.07,
                           "platform_pitch_deg": pitch,
                           "air_temperature_degC": 12.0, "relative_humidity": 0.85})
        print(f"\npitch={pitch} deg: P={float(r['power_kw']):.1f} kW, "
              f"Cp={float(r['power_coefficient']):.4f}, "
              f"pitch_factor={float(r['pitch_factor']):.4f}")
