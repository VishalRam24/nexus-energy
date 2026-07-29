"""EC110 — Reciprocating Gas Engine — F1a Part-Load Efficiency — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import ReciprocatingGasEngineF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ReciprocatingGasEngineF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            part_load_ratio : float or array — PLR [0.5–1.0]
            ambient_temp_c  : float or array — degC (optional, default 25)
        returns:
            electrical_power_kw  : kW_e
            fuel_input_kw        : kW (LHV)
            eta_electrical       : dimensionless
            gas_mass_flow_kgs    : kg/s
            gas_volume_flow_m3h  : m^3/h (at STP)
            sfc_gkwh             : g of natural gas per kWh_e
        """
        plr = np.asarray(inputs["part_load_ratio"], dtype=float)
        T_amb = np.asarray(inputs.get("ambient_temp_c", 25.0), dtype=float)
        return self._model.compute(plr, T_amb)

    def get_info(self) -> dict:
        return {
            "name": "Reciprocating Gas Engine",
            "ec_id": "EC110",
            "fidelity": "F1a",
            "description": "Part-load electrical efficiency for lean-burn NG reciprocating engine genset",
            "inputs": {
                "part_load_ratio": {"unit": "-", "range": [0.5, 1.0]},
                "ambient_temp_c":  {"unit": "degC", "range": [-20.0, 50.0], "default": 25.0},
            },
            "outputs": {
                "electrical_power_kw": {"unit": "kW_e"},
                "fuel_input_kw":       {"unit": "kW (LHV)"},
                "eta_electrical":      {"unit": "-"},
                "gas_mass_flow_kgs":   {"unit": "kg/s"},
                "gas_volume_flow_m3h": {"unit": "m^3/h (STP)"},
                "sfc_gkwh":            {"unit": "g/kWh"},
            },
            "source": "US EPA CHP Catalog (2017); Jenbacher product data",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"part_load_ratio": 1.0, "ambient_temp_c": 25.0})
    print(f"Full load: P_el={float(r['electrical_power_kw']):.0f}kW, "
          f"eta_el={float(r['eta_electrical']):.3f}, "
          f"fuel={float(r['fuel_input_kw']):.0f}kW, "
          f"gas={float(r['gas_volume_flow_m3h']):.1f} m^3/h, "
          f"SFC={float(r['sfc_gkwh']):.1f} g/kWh")
