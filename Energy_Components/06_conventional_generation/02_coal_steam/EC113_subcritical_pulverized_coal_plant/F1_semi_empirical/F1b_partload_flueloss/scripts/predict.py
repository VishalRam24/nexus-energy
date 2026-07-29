"""EC113 -- Subcritical Coal Plant -- F1b Part-Load + Flue Loss -- Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import SubcriticalCoalF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SubcriticalCoalF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            part_load_ratio : float or array [-], range [0.30, 1.0]
            ambient_temp    : float or array [degC], range [-10, 45]
        returns:
            power_mw            : net electrical output [MW_e]
            efficiency          : net LHV efficiency [-]
            coal_rate_kgs       : coal mass flow [kg/s]
            co2_rate_kgs        : CO2 emission rate [kg/s]
            co2_intensity       : CO2 intensity [g_CO2/kWh_e]
            stack_temp_c        : flue gas stack temperature [degC]
            flue_heat_loss_mw   : flue gas enthalpy loss [MW_th]
            aux_power_fraction  : auxiliary power fraction of gross output [-]
        """
        plr   = np.asarray(inputs["part_load_ratio"], dtype=float)
        T_amb = np.asarray(inputs["ambient_temp"],    dtype=float)
        return {
            "power_mw":           self._model.power_mw(plr),
            "efficiency":         self._model.efficiency_net(plr, T_amb),
            "coal_rate_kgs":      self._model.coal_rate_kgs(plr, T_amb),
            "co2_rate_kgs":       self._model.co2_rate_kgs(plr, T_amb),
            "co2_intensity":      self._model.co2_intensity_g_per_kwh(plr, T_amb),
            "stack_temp_c":       self._model.stack_temperature_c(plr),
            "flue_heat_loss_mw":  self._model.flue_heat_loss_mw(plr, T_amb),
            "aux_power_fraction": self._model.aux_power_fraction(plr),
        }

    def get_info(self) -> dict:
        return {
            "name": "Subcritical Pulverized Coal Plant",
            "ec_id": "EC113",
            "fidelity": "F1b",
            "description": (
                "Extends F1a with explicit flue gas enthalpy loss, "
                "stack temperature model, and auxiliary power part-load curve."
            ),
            "inputs": {
                "part_load_ratio": {"unit": "dimensionless", "range": [0.30, 1.0]},
                "ambient_temp":    {"unit": "degC", "range": [-10.0, 45.0]},
            },
            "outputs": {
                "power_mw":           {"unit": "MW_e"},
                "efficiency":         {"unit": "dimensionless (LHV)"},
                "coal_rate_kgs":      {"unit": "kg/s"},
                "co2_rate_kgs":       {"unit": "kg_CO2/s"},
                "co2_intensity":      {"unit": "g_CO2/kWh_e"},
                "stack_temp_c":       {"unit": "degC"},
                "flue_heat_loss_mw":  {"unit": "MW_th"},
                "aux_power_fraction": {"unit": "dimensionless"},
            },
            "source": "Booras & Holt (2004) EPRI; IEA CCC/168 (2010)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    print(f"Rated: P={float(r['power_mw']):.1f} MW, eta={float(r['efficiency']):.3f}, "
          f"T_stack={float(r['stack_temp_c']):.1f}C, "
          f"Q_flue={float(r['flue_heat_loss_mw']):.1f} MW, "
          f"CO2={float(r['co2_intensity']):.0f} g/kWh")
