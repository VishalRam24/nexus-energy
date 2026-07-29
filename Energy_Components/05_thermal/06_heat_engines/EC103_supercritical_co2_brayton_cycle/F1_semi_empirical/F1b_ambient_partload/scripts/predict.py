"""EC103 — sCO2 Brayton Cycle — F1b T_reject + Part-Load + Recuperator — Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import SCO2BraytonF1b

_PARAMS_PATH = Path(__file__).parent.parent / "data" / "parameters.json"


def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)


class ComponentModel:
    component_id = "EC103"
    component_name = "Supercritical CO2 Brayton Cycle"
    fidelity = "F1b"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            for k, v in params.items():
                if k in self._raw["cycle"]:
                    self._raw["cycle"][k]["value"] = v
        self._physics = SCO2BraytonF1b(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            PLR          : float or array [-], 0.25-1.0
            T_in         : float or array [degC], turbine inlet temperature (optional)
            T_reject     : float or array [degC], compressor inlet / heat rejection temp (optional)
        returns:
            efficiency_gross          : gross cycle efficiency [-]
            efficiency_net            : net efficiency (after aux) [-]
            power_output_mw           : net electrical output [MW]
            heat_input_mw             : thermal heat input [MW]
            heat_rejection_mw         : heat rejected [MW]
            eta_carnot                : Carnot upper bound [-]
            f_T_reject                : T_reject correction factor [-]
            f_partload                : part-load correction factor [-]
            recuperator_effectiveness : epsilon_recup at operating point [-]
        """
        PLR    = np.asarray(inputs["PLR"], dtype=float)
        T_in   = inputs.get("T_in",     None)
        T_rej  = inputs.get("T_reject",  None)
        if T_in  is not None: T_in  = np.asarray(T_in,  dtype=float)
        if T_rej is not None: T_rej = np.asarray(T_rej, dtype=float)
        return self._physics.evaluate(PLR, T_in, T_rej)

    def get_info(self) -> dict:
        return {
            "name": "Supercritical CO2 Brayton Cycle",
            "ec_id": "EC103",
            "fidelity": "F1b",
            "description": (
                "eta = eta_rated * f_PLR * f_T_reject * f_recup; "
                "critical-point penalty near T_reject=31.1 degC; "
                "recuperator effectiveness degrades at part-load."
            ),
            "inputs": {
                "PLR":      {"unit": "-",    "range": [0.25, 1.0]},
                "T_in":     {"unit": "degC", "range": [500, 800], "default": "design"},
                "T_reject": {"unit": "degC", "range": [25, 60],   "default": "design"},
            },
            "outputs": {
                "efficiency_gross":           {"unit": "-"},
                "efficiency_net":             {"unit": "-"},
                "power_output_mw":            {"unit": "MW"},
                "heat_input_mw":              {"unit": "MW"},
                "heat_rejection_mw":          {"unit": "MW"},
                "eta_carnot":                 {"unit": "-"},
                "f_T_reject":                 {"unit": "-"},
                "f_partload":                 {"unit": "-"},
                "recuperator_effectiveness":  {"unit": "-"},
            },
            "source": "Dostal et al. (2004); Crespi et al. (2017); Wright et al. (2010)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC103 F1b — sCO2 Brayton Cycle:")
    print("  Design point (T_in=700C, T_reject=32C):")
    for plr in [0.25, 0.5, 0.75, 1.0]:
        r = model.predict({"PLR": plr})
        print(f"    PLR={plr:.2f}: eta_net={float(r['efficiency_net']):.3f}, "
              f"eps_recup={float(r['recuperator_effectiveness']):.3f}")

    print("\n  T_reject sensitivity (PLR=1.0):")
    for T in [25, 31, 33, 40, 50]:
        r = model.predict({"PLR": 1.0, "T_reject": float(T)})
        print(f"    T_reject={T}C: eta_net={float(r['efficiency_net']):.3f}, "
              f"f_T={float(r['f_T_reject']):.3f}")
