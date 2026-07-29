"""EC112 — Micro Gas Turbine — F1a Part-Load Efficiency — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import MicroGasTurbineF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MicroGasTurbineF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            part_load_ratio : float or array — PLR [0.3–1.0]
            ambient_temp_c  : float or array — degC (optional, default 15)
        returns:
            electrical_power_kw  : kW_e
            fuel_input_kw        : kW (LHV)
            eta_electrical       : dimensionless
            gas_mass_flow_kgs    : kg/s
            gas_volume_flow_m3h  : m^3/h (STP)
            heat_rate_kjkwh      : kJ/kWh
        """
        plr = np.asarray(inputs["part_load_ratio"], dtype=float)
        T_amb = np.asarray(inputs.get("ambient_temp_c", 15.0), dtype=float)
        return self._model.compute(plr, T_amb)

    def get_info(self) -> dict:
        return {
            "name": "Micro Gas Turbine",
            "ec_id": "EC112",
            "fidelity": "F1a",
            "description": "Part-load electrical efficiency for recuperated micro gas turbine (30-300 kWe)",
            "inputs": {
                "part_load_ratio": {"unit": "-", "range": [0.3, 1.0]},
                "ambient_temp_c":  {"unit": "degC", "range": [-20.0, 50.0], "default": 15.0},
            },
            "outputs": {
                "electrical_power_kw": {"unit": "kW_e"},
                "fuel_input_kw":       {"unit": "kW (LHV)"},
                "eta_electrical":      {"unit": "-"},
                "gas_mass_flow_kgs":   {"unit": "kg/s"},
                "gas_volume_flow_m3h": {"unit": "m^3/h (STP)"},
                "heat_rate_kjkwh":     {"unit": "kJ/kWh"},
            },
            "source": "US EPA CHP Catalog (2017) Section 5; Capstone C200 product data",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp_c": 15.0})
    print(f"Full load ISO: P={float(r['electrical_power_kw']):.0f} kW, "
          f"eta={float(r['eta_electrical']):.3f}, "
          f"HR={float(r['heat_rate_kjkwh']):.0f} kJ/kWh, "
          f"gas={float(r['gas_volume_flow_m3h']):.1f} m^3/h")
