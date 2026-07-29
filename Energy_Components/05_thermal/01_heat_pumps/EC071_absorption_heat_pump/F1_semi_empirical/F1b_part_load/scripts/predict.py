"""EC071 — Absorption Heat Pump — F1b Part-Load — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import AbsorptionHeatPumpF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AbsorptionHeatPumpF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        T_gen   = np.asarray(inputs["T_gen"],   dtype=float)
        T_evap  = np.asarray(inputs["T_evap"],  dtype=float)
        T_cond  = np.asarray(inputs["T_cond"],  dtype=float)
        plr     = np.asarray(inputs.get("part_load_ratio", 1.0), dtype=float)
        return {
            "cop":                  self._model.cop(T_gen, T_evap, T_cond, plr),
            "heating_capacity_kw":  self._model.heating_capacity(plr),
            "driving_heat_kw":      self._model.driving_heat(T_gen, T_evap, T_cond, plr),
            "evaporator_heat_kw":   self._model.evaporator_heat(T_gen, T_evap, T_cond, plr),
            "electrical_input_kw":  self._model.electrical_input(plr),
            "cop_degradation_factor": self._model.cop_degradation_factor(T_gen, plr),
        }

    def get_info(self) -> dict:
        return {
            "name": "Absorption Heat Pump (LiBr-H2O)",
            "ec_id": "EC071",
            "fidelity": "F1b",
            "description": (
                "Part-load COP with PLF = 1 - C_d*(1-PLR), "
                "generator-T sensitivity exp(-beta*max(T_gen_des-T_gen,0)), "
                "and cycling losses below PLR_min."
            ),
            "inputs": {
                "T_gen":   {"unit": "degC", "range": [70.0, 110.0]},
                "T_evap":  {"unit": "degC", "range": [0.0,  25.0]},
                "T_cond":  {"unit": "degC", "range": [25.0, 50.0]},
                "part_load_ratio": {"unit": "-", "range": [0.0, 1.0], "default": 1.0},
            },
            "outputs": {
                "cop":                  {"unit": "dimensionless"},
                "heating_capacity_kw":  {"unit": "kW_th"},
                "driving_heat_kw":      {"unit": "kW_th"},
                "evaporator_heat_kw":   {"unit": "kW_th"},
                "electrical_input_kw":  {"unit": "kW_e"},
                "cop_degradation_factor": {"unit": "dimensionless"},
            },
            "source": "Hellmann & Ziegler (1999); Herold et al. (2016); Jakob et al. (2008)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for plr in [1.0, 0.75, 0.5, 0.25, 0.15]:
        r = model.predict({"T_gen": 90.0, "T_evap": 10.0, "T_cond": 35.0,
                           "part_load_ratio": plr})
        print(f"PLR={plr:.2f}: COP={float(r['cop']):.3f}, "
              f"Q_drive={float(r['driving_heat_kw']):.1f}kW, "
              f"degradation={float(r['cop_degradation_factor']):.3f}")
    print("\nGenerator temperature sweep (PLR=1):")
    for Tg in [75, 80, 85, 90, 95, 100]:
        r = model.predict({"T_gen": float(Tg), "T_evap": 10.0, "T_cond": 35.0})
        print(f"  T_gen={Tg}C: COP={float(r['cop']):.3f}")
