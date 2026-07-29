"""EC104 -- Gas Engine CHP -- F2a Otto Cycle -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import GasEngineCHPF2a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = GasEngineCHPF2a(self.params)

    def predict(self, inputs: dict) -> dict:
        fuel = inputs["fuel_input_kw"]
        r = inputs.get("compression_ratio", None)
        T_amb = inputs.get("T_ambient_K", 298.15)
        return self._model.compute(fuel, r, T_amb)

    def get_info(self) -> dict:
        return {
            "name": "Gas Engine CHP",
            "ec_id": "EC104",
            "fidelity": "F2a",
            "sub_fidelity": "thermo_cycle_ss",
            "description": (
                "Air-standard Otto cycle: isentropic compression + constant volume "
                "heat addition + isentropic expansion. Exhaust and jacket heat recovery."
            ),
            "inputs": {
                "fuel_input_kw": {"unit": "kW", "range": [50, 5000]},
                "compression_ratio": {"unit": "dimensionless", "default": 12},
                "T_ambient_K": {"unit": "K", "default": 298.15},
            },
            "outputs": {
                "power_electrical_kw": {"unit": "kW"},
                "heat_exhaust_kw": {"unit": "kW"},
                "heat_jacket_kw": {"unit": "kW"},
                "eta_electrical": {"unit": "dimensionless"},
                "eta_thermal": {"unit": "dimensionless"},
                "T_exhaust_K": {"unit": "K"},
            },
            "source": "Cengel & Boles (2019); US EPA (2017)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"fuel_input_kw": 500.0})
    print(f"P_elec: {r['power_electrical_kw']:.1f} kW")
    print(f"Q_exhaust: {r['heat_exhaust_kw']:.1f} kW")
    print(f"Q_jacket: {r['heat_jacket_kw']:.1f} kW")
    print(f"eta_el: {r['eta_electrical']:.3f}")
    print(f"eta_th: {r['eta_thermal']:.3f}")
    print(f"eta_total: {r['eta_total']:.3f}")
