"""EC067 — Airborne Wind Energy (AWE) — F1b Turbulence + Altitude Density — Standardized Predict Interface"""

import json
import numpy as np
from pathlib import Path
from model import AirborneWindF1b


class ComponentModel:
    """Standardized interface for EC067 AWE — F1b turbulence correction + altitude air density."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AirborneWindF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "wind_speed_m_s": m/s at operational altitude (required),
                "turbulence_intensity": 0-0.20 (optional, default 0.0),
                "altitude_m": operational altitude m (optional, uses system default ~400 m),
                "air_temperature_degC": ground-level degC (optional, default None = ISA 15degC),
                "relative_humidity": 0-1 (optional, default 0)
            }
        Returns:
            {
                "power_kw": kW,
                "capacity_factor": dimensionless,
                "air_density": kg/m3 at altitude,
                "loyd_limit_kw": kW (theoretical upper bound),
                "turbulence_correction": dimensionless
            }
        """
        v = np.asarray(inputs["wind_speed_m_s"], dtype=float)
        ti = np.asarray(inputs.get("turbulence_intensity", 0.0), dtype=float)
        alt = inputs.get("altitude_m", None)
        T = inputs.get("air_temperature_degC", None)
        RH = np.asarray(inputs.get("relative_humidity", 0.0), dtype=float)

        rho = self._model.air_density(alt, T, RH)
        p_kw = self._model.power(v, ti, alt, T, RH)

        return {
            "power_kw": p_kw,
            "capacity_factor": self._model.capacity_factor(v, ti, alt, T, RH),
            "air_density": rho,
            "loyd_limit_kw": self._model.loyd_limit_kw(v, alt, T, RH),
            "turbulence_correction": self._model.turbulence_correction(v, ti, alt, T, RH),
        }

    def get_info(self) -> dict:
        return {
            "name": "Airborne Wind Energy (AWE) System",
            "ec_id": "EC067",
            "fidelity": "F1b",
            "description": (
                "AWE power curve with turbulence correction P_corr = P(V) + 0.5*d2P/dV2*(TI*V)^2 "
                "and air-density adjustment using ISA barometric formula at operational altitude, "
                "with humidity correction (Buck 1981)."
            ),
            "inputs": {
                "wind_speed_m_s": {"unit": "m/s", "range": [0.0, 28.0],
                                    "note": "Wind speed at operational altitude"},
                "turbulence_intensity": {"unit": "dimensionless", "range": [0.0, 0.20], "default": 0.0,
                                          "note": "AWE at altitude: typically 0.05-0.12"},
                "altitude_m": {"unit": "m", "range": [100.0, 800.0], "default": 400.0,
                                "note": "Operational (kite/wing) altitude"},
                "air_temperature_degC": {"unit": "degC", "range": [-20.0, 40.0], "default": 15.0,
                                          "note": "Ground-level temperature for ISA offset"},
                "relative_humidity": {"unit": "dimensionless", "range": [0.0, 1.0], "default": 0.0},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "capacity_factor": {"unit": "dimensionless"},
                "air_density": {"unit": "kg/m3", "note": "At operational altitude"},
                "loyd_limit_kw": {"unit": "kW", "note": "Theoretical Loyd (1980) upper bound"},
                "turbulence_correction": {"unit": "dimensionless",
                                           "note": "(P_turb - P_base) / P_rated"},
            },
            "source": (
                "Loyd (1980) J. Energy 4(3); Fagiano & Milanese (2012) Automatica; "
                "Luchsinger (2013) Energies; Buck (1981) J. Appl. Meteorol.; IEC 61400-12-1"
            ),
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    for alt in [100, 400, 700]:
        r = model.predict({"wind_speed_m_s": 8.0, "turbulence_intensity": 0.08,
                           "altitude_m": alt})
        print(f"\nAlt={alt}m: P={float(r['power_kw']):.2f}kW, "
              f"rho={float(r['air_density']):.4f} kg/m3, "
              f"Loyd={float(r['loyd_limit_kw']):.2f}kW")
