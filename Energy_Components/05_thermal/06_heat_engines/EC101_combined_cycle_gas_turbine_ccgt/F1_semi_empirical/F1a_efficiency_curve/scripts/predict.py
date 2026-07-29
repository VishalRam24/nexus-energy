"""EC101 — CCGT — F1a Efficiency Curve — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import CCGTF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CCGTF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        plr   = np.asarray(inputs["part_load_ratio"], dtype=float)
        T_amb = np.asarray(inputs["ambient_temp"],     dtype=float)
        return {
            "power_mw":      self._model.power_mw(plr),
            "efficiency":    self._model.efficiency(plr, T_amb),
            "fuel_rate_kgs": self._model.fuel_rate_kgs(plr, T_amb),
            "exhaust_temp_c": self._model.exhaust_temp_c(plr),
        }

    def get_info(self) -> dict:
        return {
            "name":        "Combined Cycle Gas Turbine (CCGT)",
            "ec_id":       "EC101",
            "fidelity":    "F1a",
            "description": "eta(PLR,T_amb) = eta_iso * f_PLR(PLR) * f_amb(T_amb); quadratic part-load + linear ambient derating",
            "inputs": {
                "part_load_ratio": {"unit": "dimensionless", "range": [0.3, 1.0]},
                "ambient_temp":    {"unit": "degC",          "range": [-20.0, 50.0]},
            },
            "outputs": {
                "power_mw":       {"unit": "MW_e"},
                "efficiency":     {"unit": "dimensionless (LHV)"},
                "fuel_rate_kgs":  {"unit": "kg/s"},
                "exhaust_temp_c": {"unit": "degC"},
            },
            "source":  "Kehlhofer et al. (2009), Combined-Cycle Gas & Steam Turbine Power Plants",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    print(f"Rated ISO: P={float(r['power_mw']):.1f} MW, eta={float(r['efficiency']):.3f}, "
          f"fuel={float(r['fuel_rate_kgs']):.2f} kg/s, T_exh={float(r['exhaust_temp_c']):.1f} C")
