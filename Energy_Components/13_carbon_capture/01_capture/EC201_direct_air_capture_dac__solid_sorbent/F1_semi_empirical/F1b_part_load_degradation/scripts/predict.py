"""EC201 — DAC Solid Sorbent — F1b Part-Load Degradation — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import DACF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = DACF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict DAC performance with sorbent degradation and humidity effects.

        Parameters
        ----------
        inputs : dict
            air_flow_m3_s       : float (m3/s, default 10.0)
            T_ambient_degC      : float (degC, default 20.0)
            relative_humidity   : float (0.2-0.8, default 0.5)
            PLR                 : float or array (0.3-1.0, default 1.0)
            n_cycles            : float (default 0)
        """
        flow = inputs.get("air_flow_m3_s", 10.0)
        T = inputs.get("T_ambient_degC", 20.0)
        rh = inputs.get("relative_humidity", 0.5)
        plr = np.asarray(inputs.get("PLR", 1.0), dtype=float)
        n = inputs.get("n_cycles", 0)

        return self._model.compute(flow, T, rh, plr, n)

    def get_info(self) -> dict:
        return {
            "name": "Direct Air Capture (DAC) — Solid Sorbent",
            "ec_id": "EC201",
            "fidelity": "F1b",
            "description": (
                "Solid-sorbent DAC with temperature swing. Sorbent degradation: "
                "q(n)=q0*(1-k_deg*n), k_deg=5e-5/cycle. Humidity affects capacity. "
                "Part-load reduces air throughput."
            ),
            "inputs": {
                "air_flow_m3_s": {"unit": "m3/s", "range": [1, 100], "default": 10.0},
                "T_ambient_degC": {"unit": "degC", "range": [-10, 45], "default": 20.0},
                "relative_humidity": {"unit": "dimensionless", "range": [0.2, 0.8], "default": 0.5},
                "PLR": {"unit": "dimensionless", "range": [0.3, 1.0], "default": 1.0},
                "n_cycles": {"unit": "cycles", "range": [0, 100000], "default": 0},
            },
            "outputs": {
                "co2_captured_kg_h": {"unit": "kg/h"},
                "thermal_energy_kwh_ton": {"unit": "kWh_th/tCO2"},
                "electrical_energy_kwh_ton": {"unit": "kWh_e/tCO2"},
                "sorbent_capacity_pct": {"unit": "%"},
            },
            "source": "Fasihi et al. (2019); Sinha et al. (2017)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"air_flow_m3_s": 10.0, "T_ambient_degC": 20.0,
                        "relative_humidity": 0.5, "PLR": 1.0, "n_cycles": 0})
    print("Design point (fresh sorbent, PLR=1.0):")
    for k, v in r.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")
