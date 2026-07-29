"""EC114 — Supercritical / Ultra-Supercritical Coal Plant — F1a Efficiency Curve — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import SupercriticalCoalF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SupercriticalCoalF1a(self.params)

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
            "name":        "Supercritical / Ultra-Supercritical Coal Plant",
            "ec_id":       "EC114",
            "fidelity":    "F1a",
            "description": (
                "eta(PLR, T_amb) = eta_iso * f_PLR(PLR) * f_amb(T_amb); "
                "SC: ~600C/250bar eta~42-44%; USC: ~700C/300bar eta~44-47%; "
                "CO2 ~750-850 g/kWh"
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
            "source":  "Weitzel (2011), ASME PVP-2011-57934; IEA CCC/168 (2010); Luo et al. (2013), Energy 57",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    print(f"Rated ISO: P={float(r['power_mw']):.1f} MW, eta={float(r['efficiency'])*100:.1f}%, "
          f"coal={float(r['coal_rate_kgs']):.2f} kg/s, "
          f"CO2={float(r['co2_intensity']):.0f} g/kWh")
