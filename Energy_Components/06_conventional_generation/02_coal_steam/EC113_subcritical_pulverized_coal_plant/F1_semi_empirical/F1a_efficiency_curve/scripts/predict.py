"""EC113 — Subcritical Pulverized Coal Plant — F1a Efficiency Curve — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import SubcriticalCoalF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SubcriticalCoalF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            part_load_ratio : float or array [-], range [0.30, 1.0]
            ambient_temp    : float or array [degC], range [-10, 45]
        returns:
            power_mw          : electrical output [MW_e]
            efficiency        : net LHV efficiency [-]
            coal_rate_kgs     : coal mass flow [kg/s]
            co2_rate_kgs      : CO2 emission rate [kg/s]
            co2_intensity     : CO2 intensity [g_CO2/kWh_e]
        """
        plr   = np.asarray(inputs["part_load_ratio"], dtype=float)
        T_amb = np.asarray(inputs["ambient_temp"],     dtype=float)
        return {
            "power_mw":      self._model.power_mw(plr),
            "efficiency":    self._model.efficiency(plr, T_amb),
            "coal_rate_kgs": self._model.coal_rate_kgs(plr, T_amb),
            "co2_rate_kgs":  self._model.co2_rate_kgs(plr, T_amb),
            "co2_intensity": self._model.co2_intensity_g_per_kwh(plr, T_amb),
        }

    def get_info(self) -> dict:
        return {
            "name":        "Subcritical Pulverized Coal Plant",
            "ec_id":       "EC113",
            "fidelity":    "F1a",
            "description": (
                "eta(PLR, T_amb) = eta_iso * f_PLR(PLR) * f_amb(T_amb); "
                "linear part-load + condenser back-pressure derating; "
                "steam ~540 C / 160 bar; eta_net ~35-38%"
            ),
            "inputs": {
                "part_load_ratio": {"unit": "dimensionless", "range": [0.30, 1.0]},
                "ambient_temp":    {"unit": "degC",          "range": [-10.0, 45.0]},
            },
            "outputs": {
                "power_mw":      {"unit": "MW_e"},
                "efficiency":    {"unit": "dimensionless (LHV)"},
                "coal_rate_kgs": {"unit": "kg/s"},
                "co2_rate_kgs":  {"unit": "kg_CO2/s"},
                "co2_intensity": {"unit": "g_CO2/kWh_e"},
            },
            "source":  "Booras & Holt (2004), EPRI; IEA CCC/168 (2010)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    print(f"Rated ISO: P={float(r['power_mw']):.1f} MW, eta={float(r['efficiency']):.3f}, "
          f"coal={float(r['coal_rate_kgs']):.2f} kg/s, "
          f"CO2={float(r['co2_intensity']):.0f} g/kWh")
