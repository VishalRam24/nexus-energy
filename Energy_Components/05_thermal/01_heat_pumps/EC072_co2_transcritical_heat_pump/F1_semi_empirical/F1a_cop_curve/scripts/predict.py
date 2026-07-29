"""EC072 — CO2 Transcritical Heat Pump — F1a COP Curve — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import CO2TranscriticalHPF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CO2TranscriticalHPF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        Te  = np.asarray(inputs["T_evap"],      dtype=float)
        Twi = np.asarray(inputs["T_water_in"],  dtype=float)
        Two = np.asarray(inputs["T_water_out"], dtype=float)
        plr = np.asarray(inputs.get("part_load_ratio", 1.0), dtype=float)
        return {
            "cop":                  self._model.cop(Te, Twi, Two),
            "heating_capacity_kw":  self._model.heating_capacity(Te, Twi, Two, plr),
            "electrical_input_kw":  self._model.electrical_input(Te, Twi, Two, plr),
            "p_high_opt_bar":       self._model.optimum_high_pressure(Twi + self._model.pinch),
        }

    def get_info(self) -> dict:
        return {
            "name": "CO2 Transcritical Heat Pump (R744)",
            "ec_id": "EC072",
            "fidelity": "F1a",
            "description": "Transcritical R744 cycle with gas cooler; COP curve sensitive to T_water_out and optimum P_high",
            "inputs": {
                "T_evap":      {"unit": "degC", "range": [-20.0, 20.0]},
                "T_water_in":  {"unit": "degC", "range": [5.0, 50.0]},
                "T_water_out": {"unit": "degC", "range": [40.0, 90.0]},
                "part_load_ratio": {"unit": "-", "range": [0.0, 1.0], "default": 1.0},
            },
            "outputs": {
                "cop":                 {"unit": "dimensionless"},
                "heating_capacity_kw": {"unit": "kW_th"},
                "electrical_input_kw": {"unit": "kW_e"},
                "p_high_opt_bar":      {"unit": "bar"},
            },
            "source": "Lorentzen (1994); Sarkar et al. (2004); Liao et al. (2000)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_evap": 0.0, "T_water_in": 15.0, "T_water_out": 65.0})
    print(f"At E0/W15->65: COP={float(r['cop']):.2f}, "
          f"Q={float(r['heating_capacity_kw']):.1f}kW, "
          f"W={float(r['electrical_input_kw']):.2f}kW, "
          f"P_high_opt={float(r['p_high_opt_bar']):.1f} bar")
