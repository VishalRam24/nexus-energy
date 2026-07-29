"""EC097 — Rankine Steam Turbine — F1b Ambient + Part-Load — Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import RankineCycleF1b

_PARAMS_PATH = Path(__file__).parent.parent / "data" / "parameters.json"


def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)


class ComponentModel:
    component_id = "EC097"
    component_name = "Rankine Cycle Steam Turbine"
    fidelity = "F1b"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            for k, v in params.items():
                if k in self._raw["turbine"]:
                    self._raw["turbine"][k]["value"] = v
        self._physics = RankineCycleF1b(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            PLR          : float or array [-], 0.2-1.0
            T_condenser  : float or array [degC], optional, default=design
        returns:
            efficiency_gross     : gross cycle thermal efficiency [-]
            efficiency_net       : net efficiency (after aux power) [-]
            power_output_mw      : net electrical output [MW]
            heat_input_mw        : thermal heat input [MW]
            heat_rejection_mw    : heat rejected to condenser [MW]
            condenser_pressure_kpa: condenser saturation pressure [kPa]
            f_condenser          : condenser temperature correction factor [-]
            f_partload           : part-load correction factor [-]
        """
        PLR = np.asarray(inputs["PLR"], dtype=float)
        T_c = inputs.get("T_condenser", None)
        if T_c is not None:
            T_c = np.asarray(T_c, dtype=float)
        return self._physics.evaluate(PLR, T_c)

    def get_info(self) -> dict:
        return {
            "name": "Rankine Cycle Steam Turbine",
            "ec_id": "EC097",
            "fidelity": "F1b",
            "description": (
                "eta_gross = eta_rated*(1-a*(1-PLR)^2)*(1-k_cond*(T_cond-T_cond_design)), "
                "capped at Carnot; net = gross*(1-aux_fraction)."
            ),
            "inputs": {
                "PLR":        {"unit": "-",    "range": [0.2, 1.0]},
                "T_condenser": {"unit": "degC", "range": [15, 55], "default": "design"},
            },
            "outputs": {
                "efficiency_gross":      {"unit": "-"},
                "efficiency_net":        {"unit": "-"},
                "power_output_mw":       {"unit": "MW"},
                "heat_input_mw":         {"unit": "MW"},
                "heat_rejection_mw":     {"unit": "MW"},
                "condenser_pressure_kpa":{"unit": "kPa"},
                "f_condenser":           {"unit": "-"},
                "f_partload":            {"unit": "-"},
            },
            "source": "Cotton (1998); Spencer-Cotton-Cannon (1963); EPRI TR-107274",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC097 F1b — Rankine Steam Turbine:")
    print("  Design conditions (T_cond=33C):")
    for plr in [0.2, 0.4, 0.6, 0.8, 1.0]:
        r = model.predict({"PLR": plr})
        print(f"    PLR={plr:.1f}: eta_net={float(r['efficiency_net']):.3f}, "
              f"P_out={float(r['power_output_mw']):.1f} MW, "
              f"f_PLR={float(r['f_partload']):.3f}")
    print("\n  Condenser temperature sensitivity (PLR=1.0):")
    for T in [18, 25, 33, 40, 48]:
        r = model.predict({"PLR": 1.0, "T_condenser": T})
        print(f"    T_cond={T}C: eta_net={float(r['efficiency_net']):.3f}, "
              f"P_cond={float(r['condenser_pressure_kpa']):.2f} kPa, "
              f"f_cond={float(r['f_condenser']):.3f}")
