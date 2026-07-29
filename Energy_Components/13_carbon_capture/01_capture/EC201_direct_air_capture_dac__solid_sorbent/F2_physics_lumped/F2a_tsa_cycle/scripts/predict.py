"""
EC201 -- Direct Air Capture (DAC) Solid Sorbent -- F2a TSA Cycle -- predict interface.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import TSACycleModel

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)


class ComponentModel:
    component_id = "EC201"
    component_name = "Direct Air Capture (DAC) -- Solid Sorbent"
    fidelity = "F2a -- TSA Cycle (Lumped Physics)"
    version = "1.0.0"

    def __init__(self, params=None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = TSACycleModel(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        inputs (all optional -- defaults from parameters.json):
            T_ads_degC       : float  Adsorption temperature [degC]
            T_des_degC       : float  Desorption temperature [degC]
            P_CO2_ambient_kPa: float  Ambient CO2 partial pressure [kPa]
            P_vac_atm        : float  Vacuum pressure during desorption [atm]

        returns:
            dict with cycle metrics (see model.compute)
        """
        T_ads = inputs.get("T_ads_degC", None)
        T_des = inputs.get("T_des_degC", None)
        P_CO2 = inputs.get("P_CO2_ambient_kPa", None)
        P_vac = inputs.get("P_vac_atm", None)

        return self._model.compute(T_ads, T_des, P_CO2, P_vac)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_ads_degC":        {"unit": "degC", "range": [-10, 45],  "default": 20},
                "T_des_degC":        {"unit": "degC", "range": [60, 150],  "default": 100},
                "P_CO2_ambient_kPa": {"unit": "kPa",  "range": [0.03, 0.06], "default": 0.042},
                "P_vac_atm":         {"unit": "atm",  "range": [0.05, 1.0],  "default": 0.2},
            },
            "outputs": {
                "q_ads_mmol_g": "mmol/g",
                "q_des_mmol_g": "mmol/g",
                "working_capacity_mmol_g": "mmol/g",
                "co2_per_cycle_kg": "kg",
                "thermal_energy_per_cycle_kJ": "kJ",
                "fan_energy_per_cycle_kJ": "kJ",
                "specific_thermal_GJ_tCO2": "GJ/tCO2",
                "specific_electrical_GJ_tCO2": "GJ/tCO2",
                "total_SEC_GJ_tCO2": "GJ/tCO2",
                "productivity_kg_CO2_h": "kg/h",
                "cycle_time_s": "s",
            },
            "source": self._raw.get("unit", {}).get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({})
    print(f"CO2/cycle = {r['co2_per_cycle_kg']:.2f} kg")
    print(f"SEC_th    = {r['specific_thermal_GJ_tCO2']:.2f} GJ/tCO2")
    print(f"SEC_el    = {r['specific_electrical_GJ_tCO2']:.3f} GJ/tCO2")
    print(f"SEC_total = {r['total_SEC_GJ_tCO2']:.2f} GJ/tCO2")
    print(f"Productivity = {r['productivity_kg_CO2_h']:.2f} kg CO2/h")
