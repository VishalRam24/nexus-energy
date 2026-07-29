"""EC086 — Electric Boiler — F1b Standby Loss + Ambient — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import ElectricBoilerF1b

_PARAMS_PATH = Path(__file__).parent.parent / "data" / "parameters.json"


def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)["default_parameters"]


class ComponentModel:
    component_id = "EC086"
    component_name = "Electric Boiler / Resistance Heater"
    fidelity = "F1b"

    def __init__(self, params: dict = None):
        defaults = _load_params()
        if params:
            defaults.update(params)
        self._params = defaults
        self._physics = ElectricBoilerF1b(defaults)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            PLR          : float or array [-], 0-1
            T_ambient    : float or array [degC], optional, default=design value
            T_fluid      : float or array [degC], optional, default=design value
        returns:
            efficiency          : effective electrical-to-thermal efficiency [-]
            electrical_input_kw : total electrical input [kW]
            heat_output_kw      : net useful heat output [kW]
            standby_loss_kw     : jacket standby loss [kW]
            flue_loss_kw        : zero (no combustion) [kW]
            controls_kw         : always-on controls parasitic [kW]
        """
        PLR = np.asarray(inputs["PLR"], dtype=float)
        T_amb = inputs.get("T_ambient", None)
        T_fl  = inputs.get("T_fluid", None)
        if T_amb is not None:
            T_amb = np.asarray(T_amb, dtype=float)
        if T_fl is not None:
            T_fl = np.asarray(T_fl, dtype=float)
        return self._physics.evaluate(PLR, T_amb, T_fl)

    def get_info(self) -> dict:
        return {
            "name": "Electric Boiler / Resistance Heater",
            "ec_id": "EC086",
            "fidelity": "F1b",
            "description": (
                "eta_eff = (eta_nom*(PLR*P_rated) - UA*(T_fluid-T_amb)/1000) / P_in; "
                "no flue loss (no combustion); standby loss from thermal mass."
            ),
            "inputs": {
                "PLR":        {"unit": "-",    "range": [0.0, 1.0]},
                "T_ambient":  {"unit": "degC", "range": [-20, 45], "default": "design"},
                "T_fluid":    {"unit": "degC", "range": [40, 90],  "default": "design"},
            },
            "outputs": {
                "efficiency":          {"unit": "-"},
                "electrical_input_kw": {"unit": "kW"},
                "heat_output_kw":      {"unit": "kW"},
                "standby_loss_kw":     {"unit": "kW"},
                "flue_loss_kw":        {"unit": "kW"},
                "controls_kw":         {"unit": "kW"},
            },
            "source": "ASHRAE (2020); BS EN 12828:2012; IEA Task 44",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC086 F1b — Electric Boiler (design conditions):")
    for plr in [0.1, 0.3, 0.5, 0.8, 1.0]:
        r = model.predict({"PLR": plr})
        print(f"  PLR={plr:.1f}: eta={float(r['efficiency']):.4f}, "
              f"P_in={float(r['electrical_input_kw']):.2f} kW, "
              f"Q_out={float(r['heat_output_kw']):.2f} kW, "
              f"Q_sb={float(r['standby_loss_kw']):.3f} kW")
    print("\nAmbient sensitivity (PLR=0.3):")
    for T in [-10, 0, 10, 20, 30]:
        r = model.predict({"PLR": 0.3, "T_ambient": T})
        print(f"  T_amb={T:+3d}C: eta={float(r['efficiency']):.4f}, "
              f"Q_sb={float(r['standby_loss_kw']):.3f} kW")
