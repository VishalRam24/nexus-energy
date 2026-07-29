"""EC099 — Stirling Engine — F1b Ambient T_c + Part-Load — Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import StirlingEngineF1b

_PARAMS_PATH = Path(__file__).parent.parent / "data" / "parameters.json"


def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)


class ComponentModel:
    component_id = "EC099"
    component_name = "Stirling Engine"
    fidelity = "F1b"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            for k, v in params.items():
                if k in self._raw["engine"]:
                    self._raw["engine"][k]["value"] = v
        self._physics = StirlingEngineF1b(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            PLR          : float or array [-], 0.2-1.0
            T_hot        : float or array [degC], hot-side temperature (optional)
            T_ambient    : float or array [degC], ambient temperature (optional)
        returns:
            efficiency_gross : gross cycle efficiency [-]
            efficiency_net   : net efficiency (after aux) [-]
            power_output_w   : net electrical output [W]
            heat_input_w     : thermal input to heater [W]
            heat_rejection_w : heat rejected to cooler [W]
            eta_carnot       : Carnot efficiency upper bound [-]
            T_cold_side_c    : cold-side temperature [degC]
            f_partload       : part-load correction factor [-]
        """
        PLR    = np.asarray(inputs["PLR"], dtype=float)
        T_h    = inputs.get("T_hot", None)
        T_amb  = inputs.get("T_ambient", None)
        if T_h  is not None: T_h  = np.asarray(T_h, dtype=float)
        if T_amb is not None: T_amb = np.asarray(T_amb, dtype=float)
        return self._physics.evaluate(PLR, T_h, T_amb)

    def get_info(self) -> dict:
        return {
            "name": "Stirling Engine",
            "ec_id": "EC099",
            "fidelity": "F1b",
            "description": (
                "eta = f_carnot * (1 - T_c/T_h) * (1 - a*(1-PLR)^2); "
                "T_c = T_ambient + offset; ambient dependence included."
            ),
            "inputs": {
                "PLR":       {"unit": "-",    "range": [0.2, 1.0]},
                "T_hot":     {"unit": "degC", "range": [400, 800], "default": "design"},
                "T_ambient": {"unit": "degC", "range": [-20, 45],  "default": "design"},
            },
            "outputs": {
                "efficiency_gross": {"unit": "-"},
                "efficiency_net":   {"unit": "-"},
                "power_output_w":   {"unit": "W"},
                "heat_input_w":     {"unit": "W"},
                "heat_rejection_w": {"unit": "W"},
                "eta_carnot":       {"unit": "-"},
                "T_cold_side_c":    {"unit": "degC"},
                "f_partload":       {"unit": "-"},
            },
            "source": "Kongtragool & Wongwises (2003); Cinar et al. (2005); Thombare & Verma (2008)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC099 F1b — Stirling Engine:")
    print("  PLR sweep (T_h=650C, T_amb=25C):")
    for plr in [0.2, 0.4, 0.6, 0.8, 1.0]:
        r = model.predict({"PLR": plr, "T_ambient": 25.0})
        print(f"    PLR={plr:.1f}: eta_net={float(r['efficiency_net']):.3f}, "
              f"P={float(r['power_output_w'])/1000:.2f} kW, "
              f"T_c={float(r['T_cold_side_c']):.1f} degC")
    print("\n  Ambient sensitivity (PLR=1.0):")
    for T in [-10, 5, 20, 35]:
        r = model.predict({"PLR": 1.0, "T_ambient": T})
        print(f"    T_amb={T:+3d}C: eta_net={float(r['efficiency_net']):.3f}, "
              f"T_c={float(r['T_cold_side_c']):.1f}C, eta_carnot={float(r['eta_carnot']):.3f}")
