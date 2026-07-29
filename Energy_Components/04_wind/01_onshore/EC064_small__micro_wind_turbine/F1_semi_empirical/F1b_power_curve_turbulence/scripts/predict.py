"""EC064 — Small/Micro Wind Turbine — F1b Turbulence + Air Density — Standardized Predict Interface"""

import json
import numpy as np
from pathlib import Path
from model import SmallWindTurbineF1b


class ComponentModel:
    """Standardized interface for EC064 Small/Micro Wind Turbine — F1b."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SmallWindTurbineF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "wind_speed_m_s": m/s (required),
                "turbulence_intensity": 0-0.40 (optional, default 0.0),
                "pressure_pa": Pa (optional, default 101325),
                "air_temperature_degC": degC (optional, default 15),
                "relative_humidity": 0-1 (optional, default 0)
            }
        Returns:
            {
                "power_kw": kW,
                "power_coefficient": dimensionless,
                "air_density": kg/m3,
                "turbulence_correction": dimensionless (fractional CF correction)
            }
        """
        v = np.asarray(inputs["wind_speed_m_s"], dtype=float)
        ti = np.asarray(inputs.get("turbulence_intensity", 0.0), dtype=float)
        P = np.asarray(inputs.get("pressure_pa", 101325.0), dtype=float)
        T = np.asarray(inputs.get("air_temperature_degC", 15.0), dtype=float)
        RH = np.asarray(inputs.get("relative_humidity", 0.0), dtype=float)

        rho = self._model.air_density(P, T, RH)

        return {
            "power_kw": self._model.power(v, ti, P, T, RH),
            "power_coefficient": self._model.power_coefficient(v, ti, P, T, RH),
            "air_density": rho,
            "turbulence_correction": self._model.turbulence_correction(v, ti, P, T, RH),
        }

    def get_info(self) -> dict:
        return {
            "name": "Small/Micro Wind Turbine",
            "ec_id": "EC064",
            "fidelity": "F1b",
            "description": (
                "Power curve with turbulence correction P_corr = P(V) + 0.5*d2P/dV2*(TI*V)^2 "
                "and full air-density adjustment for temperature, pressure, and humidity."
            ),
            "inputs": {
                "wind_speed_m_s": {"unit": "m/s", "range": [0.0, 25.0]},
                "turbulence_intensity": {"unit": "dimensionless", "range": [0.0, 0.40], "default": 0.0,
                                         "note": "Small-wind sites commonly 0.15-0.40"},
                "pressure_pa": {"unit": "Pa", "range": [70000.0, 110000.0], "default": 101325.0,
                                "note": "Relevant for elevated/rooftop installations"},
                "air_temperature_degC": {"unit": "degC", "range": [-20.0, 50.0], "default": 15.0},
                "relative_humidity": {"unit": "dimensionless", "range": [0.0, 1.0], "default": 0.0},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "power_coefficient": {"unit": "dimensionless"},
                "air_density": {"unit": "kg/m3"},
                "turbulence_correction": {"unit": "dimensionless",
                                          "note": "(P_turb - P_base) / P_rated"},
            },
            "source": "IEC 61400-2:2013; IEC 61400-12-1:2017; Albers et al. (2007) Wind Energy 10(4); Buck (1981)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    cases = [
        {"wind_speed_m_s": 7.0, "turbulence_intensity": 0.20},
        {"wind_speed_m_s": 7.0, "turbulence_intensity": 0.20, "pressure_pa": 90000.0,
         "air_temperature_degC": 25.0},
    ]
    for c in cases:
        r = model.predict(c)
        print(f"\nInputs: {c}")
        for k, v in r.items():
            print(f"  {k}: {float(v):.4f}")
