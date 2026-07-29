"""EC055 — Solar Tower — F1b With Thermal Losses — Standardized Predict Interface"""

import json
import numpy as np
from pathlib import Path
from model import SolarTowerF1b


class ComponentModel:
    """Standardized interface for EC055 Solar Tower — F1b thermal loss model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SolarTowerF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "dni_w_m2": W/m2 (0-1100),
                "solar_zenith_deg": degrees (0-85),
                "T_receiver_degC": degC (200-800),
                "T_ambient_degC": degC (-15 to 55),
                "wind_speed_m_s": m/s (0-20) [optional, default 0]
            }
        Returns:
            Q_field_kw, useful_heat_kw, thermal_loss_kw, Q_radiative_kw,
            Q_convective_kw, optical_efficiency, receiver_efficiency,
            overall_efficiency, h_conv_w_m2k
        """
        dni = np.asarray(inputs["dni_w_m2"], dtype=float)
        zenith = np.asarray(inputs["solar_zenith_deg"], dtype=float)
        T_recv = np.asarray(inputs["T_receiver_degC"], dtype=float)
        T_amb = np.asarray(inputs["T_ambient_degC"], dtype=float)
        wind = float(inputs.get("wind_speed_m_s", 0.0))
        return self._model.predict_all(dni, zenith, T_recv, T_amb, wind)

    def get_info(self) -> dict:
        return {
            "name": "Solar Tower Central Receiver CSP",
            "ec_id": "EC055",
            "fidelity": "F1b",
            "description": (
                "Heliostat field optical efficiency + explicit receiver thermal losses. "
                "F1b extends F1a with: (1) T_amb as explicit variable input, "
                "(2) wind-dependent convective loss h=h_base+h_wind*sqrt(v), "
                "(3) radiative loss Q_rad = eps*sigma*A*(T_r^4 - T_amb^4). "
                "Enables seasonal T_amb and weather sensitivity analysis."
            ),
            "inputs": {
                "dni_w_m2":         {"unit": "W/m2",   "range": [0.0, 1100.0]},
                "solar_zenith_deg": {"unit": "deg",    "range": [0.0, 85.0]},
                "T_receiver_degC":  {"unit": "degC",   "range": [200.0, 800.0]},
                "T_ambient_degC":   {"unit": "degC",   "range": [-15.0, 55.0]},
                "wind_speed_m_s":   {"unit": "m/s",    "range": [0.0, 20.0], "default": 0.0},
            },
            "outputs": {
                "Q_field_kw":          {"unit": "kW"},
                "useful_heat_kw":      {"unit": "kW"},
                "thermal_loss_kw":     {"unit": "kW"},
                "Q_radiative_kw":      {"unit": "kW"},
                "Q_convective_kw":     {"unit": "kW"},
                "optical_efficiency":  {"unit": "dimensionless"},
                "receiver_efficiency": {"unit": "dimensionless"},
                "overall_efficiency":  {"unit": "dimensionless"},
                "h_conv_w_m2k":        {"unit": "W/m2K"},
            },
            "source": "Wagner & Wendelin (2018); Falcone (1986) SAND86-8009; Siebers & Kraabel (1984) SAND84-8717",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"dni_w_m2": 900.0, "solar_zenith_deg": 30.0,
                       "T_receiver_degC": 600.0, "T_ambient_degC": 25.0})
    print(f"\nAt DNI=900, zenith=30, T_recv=600C, T_amb=25C:")
    for k, v in r.items():
        print(f"  {k}: {float(np.asarray(v)):.3f}")
