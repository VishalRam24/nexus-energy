"""EC111 — Diesel Generator — F1a Willans Line — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import DieselGeneratorF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = DieselGeneratorF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            power_output_kw  : float or array, generator output [kW]
            ambient_temp_c   : float or array, ambient temperature [degC] (optional, default 25)
        returns:
            fuel_rate_lph    : fuel consumption [L/h]
            sfc_gkwh         : specific fuel consumption [g/kWh]
            efficiency       : generator efficiency [-]
            co2_emissions_kgh: CO2 emission rate [kg_CO2/h]
        """
        P = np.asarray(inputs["power_output_kw"], dtype=float)
        T = np.asarray(inputs.get("ambient_temp_c", 25.0), dtype=float)
        return {
            "fuel_rate_lph": self._model.fuel_rate(P, T),
            "sfc_gkwh": self._model.sfc(P, T),
            "efficiency": self._model.efficiency(P, T),
            "co2_emissions_kgh": self._model.co2_emissions(P, T),
        }

    def get_info(self) -> dict:
        return {
            "name": "Diesel Generator",
            "ec_id": "EC111",
            "fidelity": "F1a",
            "model": "Willans Line",
            "description": "fuel_rate = a + b * P_out; eta = P_out / (fuel_rate * rho * LHV / 3.6)",
            "inputs": {
                "power_output_kw": {"unit": "kW", "range": [0.0, 500.0]},
                "ambient_temp_c": {"unit": "degC", "range": [-20.0, 55.0], "default": 25.0},
            },
            "outputs": {
                "fuel_rate_lph": {"unit": "L/h"},
                "sfc_gkwh": {"unit": "g/kWh"},
                "efficiency": {"unit": "dimensionless"},
                "co2_emissions_kgh": {"unit": "kg_CO2/h"},
            },
            "source": "US Army TM 5-811-6 (1996); Tuffaha & Gravdahl (2014)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"power_output_kw": 500.0})
    print(f"At rated load (500 kW):")
    print(f"  Fuel rate : {float(r['fuel_rate_lph']):.1f} L/h")
    print(f"  SFC       : {float(r['sfc_gkwh']):.1f} g/kWh")
    print(f"  Efficiency: {float(r['efficiency'])*100:.1f} %")
    print(f"  CO2       : {float(r['co2_emissions_kgh']):.1f} kg/h")
