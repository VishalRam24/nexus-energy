"""EC078 — Hot Water Tank TES — F1b Stratified — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import HotWaterTankF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = HotWaterTankF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        T_inlet_hot = inputs.get("T_inlet_hot", 80.0)
        T_inlet_cold = inputs.get("T_inlet_cold", 15.0)
        flow_charge = inputs.get("flow_rate_charge", 0.0)
        flow_discharge = inputs.get("flow_rate_discharge", 0.0)
        T_ambient = inputs.get("T_ambient", 20.0)
        duration_s = inputs.get("duration_s", 3600.0)
        T_initial = inputs.get("T_initial", None)
        dt = inputs.get("dt", 10.0)

        result = self._model.simulate(
            T_inlet_hot, T_inlet_cold,
            flow_charge, flow_discharge,
            T_ambient, duration_s,
            T_initial=T_initial, dt=dt,
        )

        return {
            "T_nodes": result["T_nodes"].tolist(),
            "T_outlet_hot": float(result["T_outlet_hot"]),
            "T_outlet_cold": float(result["T_outlet_cold"]),
            "stored_energy_kwh": float(result["stored_energy_kwh"]),
            "stratification_efficiency": float(result["stratification_efficiency"]),
        }

    def get_info(self) -> dict:
        return {
            "name": "Sensible Heat Storage — Hot Water Tank (stratified)",
            "ec_id": "EC078",
            "fidelity": "F1b",
            "description": "N-node stratification: dT_i/dt = f(Q_in, Q_out, UA, k_mix, advection, buoyancy)",
            "inputs": {
                "T_inlet_hot": {"unit": "degC", "range": [40.0, 95.0], "default": 80.0},
                "T_inlet_cold": {"unit": "degC", "range": [5.0, 30.0], "default": 15.0},
                "flow_rate_charge": {"unit": "kg/s", "range": [0.0, 2.0], "default": 0.0},
                "flow_rate_discharge": {"unit": "kg/s", "range": [0.0, 2.0], "default": 0.0},
                "T_ambient": {"unit": "degC", "range": [-10.0, 40.0], "default": 20.0},
                "duration_s": {"unit": "s", "range": [1.0, 86400.0], "default": 3600.0},
            },
            "outputs": {
                "T_nodes": {"unit": "degC", "note": "Array of N node temperatures (top to bottom)"},
                "T_outlet_hot": {"unit": "degC"},
                "T_outlet_cold": {"unit": "degC"},
                "stored_energy_kwh": {"unit": "kWh"},
                "stratification_efficiency": {"unit": "dimensionless"},
            },
            "source": "Duffie & Beckman (2013) ch.8; TRNSYS Type 60",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    # Charge for 1 hour, then discharge for 1 hour
    r_charge = model.predict({
        "T_inlet_hot": 80.0, "T_inlet_cold": 15.0,
        "flow_rate_charge": 0.1, "flow_rate_discharge": 0.0,
        "T_ambient": 20.0, "duration_s": 3600.0,
    })
    print(f"After 1h charge: T_nodes={[f'{t:.1f}' for t in r_charge['T_nodes']]}")
    print(f"  Stored={r_charge['stored_energy_kwh']:.2f} kWh, "
          f"Strat_eff={r_charge['stratification_efficiency']:.2f}")

    r_discharge = model.predict({
        "T_inlet_hot": 80.0, "T_inlet_cold": 15.0,
        "flow_rate_charge": 0.0, "flow_rate_discharge": 0.1,
        "T_ambient": 20.0, "duration_s": 3600.0,
        "T_initial": r_charge["T_nodes"],
    })
    print(f"After 1h discharge: T_nodes={[f'{t:.1f}' for t in r_discharge['T_nodes']]}")
    print(f"  T_outlet_hot={r_discharge['T_outlet_hot']:.1f}C")
