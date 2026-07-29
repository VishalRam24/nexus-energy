"""EC072 — CO2 Transcritical HP — F1b Part-Load — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import CO2TranscriticalHPF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CO2TranscriticalHPF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        Te   = np.asarray(inputs["T_evap"],      dtype=float)
        Twin = np.asarray(inputs["T_water_in"],  dtype=float)
        Twout= np.asarray(inputs["T_water_out"], dtype=float)
        plr  = np.asarray(inputs.get("part_load_ratio", 1.0), dtype=float)
        return {
            "cop":                  self._model.cop(Te, Twin, Twout, plr),
            "heating_capacity_kw":  self._model.heating_capacity(plr),
            "electrical_input_kw":  self._model.electrical_input(Te, Twin, Twout, plr),
            "cop_degradation_factor": self._model.cop_degradation_factor(Twin, plr),
            "optimum_P_high_bar":   self._model.optimum_high_pressure(Twin + 3.0),
        }

    def get_info(self) -> dict:
        return {
            "name": "CO2 Transcritical Heat Pump",
            "ec_id": "EC072",
            "fidelity": "F1b",
            "description": (
                "Part-load COP with PLF = 1 - C_d*(1-PLR), "
                "T_water_in penalty exp(-gamma*max(T_w_in-T_design,0)), "
                "and cycling losses below PLR_min."
            ),
            "inputs": {
                "T_evap":      {"unit": "degC", "range": [-20.0, 20.0]},
                "T_water_in":  {"unit": "degC", "range": [5.0, 50.0]},
                "T_water_out": {"unit": "degC", "range": [40.0, 90.0]},
                "part_load_ratio": {"unit": "-", "range": [0.0, 1.0], "default": 1.0},
            },
            "outputs": {
                "cop":                  {"unit": "dimensionless"},
                "heating_capacity_kw":  {"unit": "kW_th"},
                "electrical_input_kw":  {"unit": "kW_e"},
                "cop_degradation_factor": {"unit": "dimensionless"},
                "optimum_P_high_bar":   {"unit": "bar"},
            },
            "source": "Lorentzen (1994); Sarkar et al. (2004); Liao et al. (2000)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("Part-load sweep (T_evap=0C, T_w_in=15C, T_w_out=65C):")
    for plr in [1.0, 0.75, 0.5, 0.25, 0.1]:
        r = model.predict({"T_evap": 0.0, "T_water_in": 15.0,
                           "T_water_out": 65.0, "part_load_ratio": plr})
        print(f"  PLR={plr:.2f}: COP={float(r['cop']):.3f}, "
              f"W={float(r['electrical_input_kw']):.2f}kW, "
              f"degradation={float(r['cop_degradation_factor']):.3f}")
    print("\nT_water_in sweep (PLR=1, T_evap=0C, T_w_out=65C):")
    for Twin in [5, 15, 25, 35, 45]:
        r = model.predict({"T_evap": 0.0, "T_water_in": float(Twin),
                           "T_water_out": 65.0, "part_load_ratio": 1.0})
        print(f"  T_w_in={Twin}C: COP={float(r['cop']):.3f}")
