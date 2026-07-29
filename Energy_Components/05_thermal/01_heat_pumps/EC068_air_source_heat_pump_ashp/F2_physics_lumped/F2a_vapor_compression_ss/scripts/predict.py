"""EC068 — ASHP — F2a Vapor Compression SS — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import ASHPF2a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ASHPF2a(self.params)

    def predict(self, inputs: dict) -> dict:
        T_evap = float(inputs["T_evap_degC"])
        T_cond = float(inputs["T_cond_degC"])
        superheat = inputs.get("superheat_K", None)
        subcool = inputs.get("subcool_K", None)
        return self._model.solve_cycle(T_evap, T_cond, superheat, subcool)

    def get_info(self) -> dict:
        return {
            "name": "Air-Source Heat Pump (ASHP)",
            "ec_id": "EC068",
            "fidelity": "F2a",
            "model_type": "Steady-state vapor compression cycle",
            "description": "4-component vapor compression cycle (R410A) with CoolProp properties",
            "inputs": {
                "T_evap_degC": {"unit": "degC", "range": [-25, 20], "description": "Evaporating temperature"},
                "T_cond_degC": {"unit": "degC", "range": [25, 65], "description": "Condensing temperature"},
                "superheat_K": {"unit": "K", "default": 5.0, "description": "Evaporator outlet superheat"},
                "subcool_K": {"unit": "K", "default": 3.0, "description": "Condenser outlet subcooling"},
            },
            "outputs": {
                "cop": {"unit": "dimensionless"},
                "heating_capacity_kw": {"unit": "kW"},
                "compressor_power_kw": {"unit": "kW"},
                "mass_flow_kg_s": {"unit": "kg/s"},
                "state_points": {"unit": "dict", "description": "Thermodynamic state at each cycle point"},
            },
            "refrigerant": "R410A",
            "source": "ASHRAE Handbook; Cengel & Boles Thermodynamics",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_evap_degC": 0.0, "T_cond_degC": 45.0})
    print(f"COP={r['cop']:.2f}, Q_heat={r['heating_capacity_kw']:.1f}kW, "
          f"W_comp={r['compressor_power_kw']:.2f}kW, m_dot={r['mass_flow_kg_s']:.4f}kg/s")
