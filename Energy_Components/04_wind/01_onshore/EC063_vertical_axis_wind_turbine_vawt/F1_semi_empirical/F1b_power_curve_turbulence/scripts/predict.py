"""EC063 — VAWT — F1b Turbulence + Air Density — Standardized Predict Interface"""

import json
import numpy as np
from pathlib import Path
from model import VAWTF1b


class ComponentModel:
    """Standardized interface for EC063 VAWT — F1b turbulence + air density."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = VAWTF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "wind_speed_m_s": m/s,
                "turbulence_intensity": 0–0.50 (optional, default 0.0),
                "air_temperature_degC": degC (optional, default 15.0),
                "altitude_m": m (optional, default 0.0)
            }
        Returns:
            {
                "power_kw": kW,
                "power_coefficient": dimensionless,
                "capacity_factor": dimensionless,
                "air_density": kg/m3,
                "ti_modifier": dimensionless
            }
        """
        v = np.asarray(inputs["wind_speed_m_s"], dtype=float)
        ti = np.asarray(inputs.get("turbulence_intensity", 0.0), dtype=float)
        T = np.asarray(inputs.get("air_temperature_degC", 15.0), dtype=float)
        z = np.asarray(inputs.get("altitude_m", 0.0), dtype=float)

        rho = self._model.air_density(T, z)
        ti_mod = self._model.ti_modifier(v, ti)

        return {
            "power_kw": self._model.power(v, ti, T, z),
            "power_coefficient": self._model.power_coefficient(v, ti, T, z),
            "capacity_factor": self._model.capacity_factor(v, ti, T, z),
            "air_density": rho,
            "ti_modifier": ti_mod,
        }

    def get_info(self) -> dict:
        return {
            "name": "Vertical Axis Wind Turbine (VAWT)",
            "ec_id": "EC063",
            "fidelity": "F1b",
            "description": (
                "Power curve with air density rho(T, altitude) + TI effect on effective Cp: "
                "Cp_eff = Cp_base * (1 + k_ti * TI) below rated."
            ),
            "inputs": {
                "wind_speed_m_s": {"unit": "m/s", "range": [0.0, 30.0]},
                "turbulence_intensity": {"unit": "dimensionless", "range": [0.0, 0.50], "default": 0.0},
                "air_temperature_degC": {"unit": "degC", "range": [-20.0, 50.0], "default": 15.0},
                "altitude_m": {"unit": "m", "range": [0.0, 3000.0], "default": 0.0},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "power_coefficient": {"unit": "dimensionless"},
                "capacity_factor": {"unit": "dimensionless"},
                "air_density": {"unit": "kg/m3"},
                "ti_modifier": {"unit": "dimensionless"},
            },
            "source": "Tjiu et al. (2015); Simão Ferreira et al. (2007); ISO 2533",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    for ti in [0.0, 0.10, 0.25]:
        r = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": ti,
                           "air_temperature_degC": 15.0, "altitude_m": 500.0})
        print(f"\nTI={ti:.2f}: P={float(r['power_kw']):.2f} kW, "
              f"rho={float(r['air_density']):.4f} kg/m3, "
              f"ti_mod={float(r['ti_modifier']):.3f}")
