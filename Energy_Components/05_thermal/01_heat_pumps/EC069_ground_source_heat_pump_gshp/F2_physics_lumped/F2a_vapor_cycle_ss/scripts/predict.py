"""EC069 — GSHP — F2a Vapor Cycle SS — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import GSHPF2a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = GSHPF2a(self.params)

    def predict(self, inputs: dict) -> dict:
        T_cond = float(inputs["T_cond_degC"])
        Q_demand = inputs.get("Q_demand_kw", None)
        if Q_demand is not None:
            Q_demand = float(Q_demand)
        return self._model.solve_cycle(T_cond, Q_demand)

    def get_info(self) -> dict:
        return {
            "name": "Ground-Source Heat Pump (GSHP)",
            "ec_id": "EC069",
            "fidelity": "F2a",
            "model_type": "Steady-state vapor compression cycle with ground-loop coupling",
            "description": "Vapor compression cycle (R410A) coupled to vertical borehole heat exchanger",
            "inputs": {
                "T_cond_degC": {"unit": "degC", "range": [25, 55], "description": "Condensing temperature"},
                "Q_demand_kw": {"unit": "kW", "default": 15.0, "description": "Heating demand"},
            },
            "outputs": {
                "cop": {"unit": "dimensionless"},
                "heating_capacity_kw": {"unit": "kW"},
                "compressor_power_kw": {"unit": "kW"},
                "T_evap_degC": {"unit": "degC", "description": "Self-consistent evaporator temperature"},
                "T_source_degC": {"unit": "degC", "description": "Brine source temperature"},
            },
            "refrigerant": "R410A",
            "source": "Kavanaugh & Rafferty (2014); ASHRAE Handbook",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_cond_degC": 45.0})
    print(f"COP={r['cop']:.2f}, T_evap={r['T_evap_degC']:.1f}C, "
          f"W={r['compressor_power_kw']:.2f}kW, T_source={r['T_source_degC']:.1f}C")
