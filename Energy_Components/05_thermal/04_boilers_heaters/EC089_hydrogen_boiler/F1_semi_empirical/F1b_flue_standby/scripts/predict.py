"""EC089 — Hydrogen Boiler — F1b H2O-rich Flue + Condensing — Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import HydrogenBoilerF1b

_PARAMS_PATH = Path(__file__).parent.parent / "data" / "parameters.json"


def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)["default_parameters"]


class ComponentModel:
    component_id = "EC089"
    component_name = "Hydrogen Boiler"
    fidelity = "F1b"

    def __init__(self, params: dict = None):
        defaults = _load_params()
        if params:
            defaults.update(params)
        self._params = defaults
        self._physics = HydrogenBoilerF1b(defaults)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            PLR              : float or array [-], 0-1
            T_flue_override  : float or None [degC], optional
        returns:
            efficiency        : part-load thermal efficiency, LHV basis [-]
            heat_output_kw    : useful heat output [kW]
            fuel_input_kw     : H2 fuel thermal input, LHV [kW]
            flue_loss_kw      : H2O-rich flue sensible heat loss [kW]
            latent_recovery_kw: latent heat recovered (condensing only) [kW]
            standby_loss_kw   : standby thermal loss [kW]
            h2_flow_kg_s      : hydrogen mass flow [kg/s]
            flue_gas_temp_c   : flue exit temperature [degC]
            condensing        : 1.0 = condensing, 0.0 = non-condensing
        """
        PLR = np.asarray(inputs["PLR"], dtype=float)
        T_fl = inputs.get("T_flue_override", None)
        if T_fl is not None:
            T_fl = np.asarray(T_fl, dtype=float)
        return self._physics.evaluate(PLR, T_fl)

    def get_info(self) -> dict:
        return {
            "name": "Hydrogen Boiler",
            "ec_id": "EC089",
            "fidelity": "F1b",
            "description": (
                "eta_LHV(PLR) = a0 + a1*PLR + a2*PLR^2; "
                "H2O-rich flue gas loss; condensing vs non-condensing mode; standby loss."
            ),
            "inputs": {
                "PLR":             {"unit": "-",    "range": [0.0, 1.0]},
                "T_flue_override": {"unit": "degC", "range": [30, 200], "default": "auto"},
            },
            "outputs": {
                "efficiency":          {"unit": "-"},
                "heat_output_kw":      {"unit": "kW"},
                "fuel_input_kw":       {"unit": "kW"},
                "flue_loss_kw":        {"unit": "kW"},
                "latent_recovery_kw":  {"unit": "kW"},
                "standby_loss_kw":     {"unit": "kW"},
                "h2_flow_kg_s":        {"unit": "kg/s"},
                "flue_gas_temp_c":     {"unit": "degC"},
                "condensing":          {"unit": "bool"},
            },
            "source": "Hy4Heat WP6 (2021); Cellek & Pinarbasi (2018); Woolley et al. (2022)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    print("EC089 F1b — Hydrogen Boiler (condensing):")
    model_cond = ComponentModel({"condensing": True})
    model_ncond = ComponentModel({"condensing": False})
    for plr in [0.1, 0.3, 0.5, 0.8, 1.0]:
        rc = model_cond.predict({"PLR": plr})
        rn = model_ncond.predict({"PLR": plr})
        print(f"  PLR={plr:.1f}: eta_cond={float(rc['efficiency']):.3f}, "
              f"Q_flue_cond={float(rc['flue_loss_kw']):.2f} kW, "
              f"Q_lat={float(rc['latent_recovery_kw']):.2f} kW | "
              f"eta_ncond={float(rn['efficiency']):.3f}, "
              f"Q_flue_ncond={float(rn['flue_loss_kw']):.2f} kW")
