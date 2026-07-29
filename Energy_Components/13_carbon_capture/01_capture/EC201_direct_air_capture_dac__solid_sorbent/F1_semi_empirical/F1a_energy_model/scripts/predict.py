"""EC201 — DAC Solid Sorbent — F1a Energy Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import DACF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = DACF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            air_flow_m3h       : float or array — air throughput [m3/hr]
            relative_humidity  : float or array — RH [0–1]
            ambient_temp       : float or array — ambient temperature [degC], default 15
        returns:
            co2_captured_tpa          : tonnes CO2 / year
            thermal_energy_mwh_pa     : MWh_th / year
            electrical_energy_mwh_pa  : MWh_e / year
            specific_thermal_kwht     : kWh_th / tCO2
            specific_electric_kwhe    : kWh_e / tCO2
        """
        air_flow = np.asarray(inputs["air_flow_m3h"], dtype=float)
        rh = np.asarray(inputs["relative_humidity"], dtype=float)
        t_amb = np.asarray(inputs.get("ambient_temp", 15.0), dtype=float)
        return self._model.annual_outputs(air_flow, rh, t_amb)

    def get_info(self) -> dict:
        return {
            "name": "Direct Air Capture (DAC) — Solid Sorbent",
            "ec_id": "EC201",
            "fidelity": "F1a",
            "description": "Energy model: E_thermal = E_th_base / (eta_cap * humidity_factor), E_electric constant",
            "inputs": {
                "air_flow_m3h":      {"unit": "m3/hr",  "range": [1e5, 1e7]},
                "relative_humidity": {"unit": "-",      "range": [0.1, 0.9]},
                "ambient_temp":      {"unit": "degC",   "range": [-10.0, 45.0], "default": 15.0},
            },
            "outputs": {
                "co2_captured_tpa":         {"unit": "tCO2/yr"},
                "thermal_energy_mwh_pa":    {"unit": "MWh_th/yr"},
                "electrical_energy_mwh_pa": {"unit": "MWh_e/yr"},
                "specific_thermal_kwht":    {"unit": "kWh_th/tCO2"},
                "specific_electric_kwhe":   {"unit": "kWh_e/tCO2"},
            },
            "source": "Fasihi et al. (2019), J. Cleaner Production, 224, 957-980",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"air_flow_m3h": 1e6, "relative_humidity": 0.5})
    print(f"CO2 captured: {float(r['co2_captured_tpa']):.1f} tCO2/yr")
    print(f"Specific thermal: {float(r['specific_thermal_kwht']):.0f} kWh_th/tCO2")
    print(f"Specific electric: {float(r['specific_electric_kwhe']):.0f} kWh_e/tCO2")
