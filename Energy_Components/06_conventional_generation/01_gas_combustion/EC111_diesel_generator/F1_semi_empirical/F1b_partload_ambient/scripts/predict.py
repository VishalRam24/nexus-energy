"""EC111 -- Diesel Generator -- F1b Part-Load + Ambient -- Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import DieselGeneratorF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = DieselGeneratorF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            PLR          : float or array [0.25 - 1.0]
            T_ambient    : float or array [degC] (default 25.0)
            altitude_m   : float or array [m] (default 0.0)
            rated_power_kw : float (optional override)
        returns:
            efficiency          : generator efficiency [-]
            power_output_kw     : electrical output [kW]
            fuel_consumption_l_h: fuel consumption [L/h]
            sfc_g_kwh           : specific fuel consumption [g/kWh]
            exhaust_temp_degC   : exhaust temperature [degC]
        """
        PLR  = np.asarray(inputs["PLR"], dtype=float)
        T    = np.asarray(inputs.get("T_ambient", 25.0), dtype=float)
        alt  = np.asarray(inputs.get("altitude_m", 0.0), dtype=float)

        if "rated_power_kw" in inputs:
            self._model.P_rated = float(inputs["rated_power_kw"])

        return {
            "efficiency":           self._model.efficiency(PLR, T, alt),
            "power_output_kw":      self._model.power_output_kw(PLR, T, alt),
            "fuel_consumption_l_h": self._model.fuel_consumption_l_h(PLR, T, alt),
            "sfc_g_kwh":            self._model.sfc_g_kwh(PLR, T, alt),
            "exhaust_temp_degC":    self._model.exhaust_temp_c(PLR),
        }

    def get_info(self) -> dict:
        return {
            "name": "Diesel Generator",
            "ec_id": "EC111",
            "fidelity": "F1b",
            "model": "Part-Load + Ambient (Altitude & Temperature Derating)",
            "description": (
                "Willans line fuel model with altitude derating (3.5%/300m above 1000m) "
                "and temperature derating (0.5%/degC above 40C)"
            ),
            "inputs": {
                "PLR":            {"unit": "-", "range": [0.25, 1.0]},
                "T_ambient":      {"unit": "degC", "range": [-30, 55], "default": 25.0},
                "altitude_m":     {"unit": "m", "range": [0, 5000], "default": 0.0},
                "rated_power_kw": {"unit": "kW", "range": [50, 5000], "default": 500.0},
            },
            "outputs": {
                "efficiency":           {"unit": "-"},
                "power_output_kw":      {"unit": "kW"},
                "fuel_consumption_l_h": {"unit": "L/h"},
                "sfc_g_kwh":            {"unit": "g/kWh"},
                "exhaust_temp_degC":    {"unit": "degC"},
            },
            "source": "US Army TM 5-811-6; Caterpillar App Guide; ISO 8528-1",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC111 F1b -- Standard conditions (PLR=1.0, 25C, sea level):")
    r = model.predict({"PLR": 1.0})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
    print("\nHigh altitude, hot (PLR=0.75, 45C, 3000m):")
    r = model.predict({"PLR": 0.75, "T_ambient": 45.0, "altitude_m": 3000.0})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
