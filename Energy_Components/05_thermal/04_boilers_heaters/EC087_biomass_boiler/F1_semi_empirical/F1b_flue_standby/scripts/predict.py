"""EC087 — Biomass Boiler — F1b Flue Gas + Moisture + Cycling Standby — Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import BiomassBOilerF1b

_PARAMS_PATH = Path(__file__).parent.parent / "data" / "parameters.json"


def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)["default_parameters"]


class ComponentModel:
    component_id = "EC087"
    component_name = "Biomass Boiler"
    fidelity = "F1b"

    def __init__(self, params: dict = None):
        defaults = _load_params()
        if params:
            defaults.update(params)
        self._params = defaults
        self._physics = BiomassBOilerF1b(defaults)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            PLR              : float or array [-], 0-1
            T_flue_override  : float or None [degC], optional flue temperature override
        returns:
            efficiency       : part-load thermal efficiency [-]
            heat_output_kw   : useful heat output [kW]
            fuel_input_kw    : fuel thermal input, LHV [kW]
            flue_loss_kw     : flue gas sensible heat loss [kW]
            standby_loss_kw  : constant casing standby loss [kW]
            cycling_loss_kw  : cycling start/stop loss [kW]
            flue_gas_temp_c  : flue gas temperature [degC]
        """
        PLR = np.asarray(inputs["PLR"], dtype=float)
        T_fl = inputs.get("T_flue_override", None)
        if T_fl is not None:
            T_fl = np.asarray(T_fl, dtype=float)
        return self._physics.evaluate(PLR, T_fl)

    def get_info(self) -> dict:
        return {
            "name": "Biomass Boiler",
            "ec_id": "EC087",
            "fidelity": "F1b",
            "description": (
                "eta(PLR) = a0 + a1*PLR + a2*PLR^2; "
                "flue loss with excess-air + moisture correction; "
                "standby + cycling losses."
            ),
            "inputs": {
                "PLR":             {"unit": "-",    "range": [0.0, 1.0]},
                "T_flue_override": {"unit": "degC", "range": [50, 300], "default": "auto"},
            },
            "outputs": {
                "efficiency":      {"unit": "-"},
                "heat_output_kw":  {"unit": "kW"},
                "fuel_input_kw":   {"unit": "kW"},
                "flue_loss_kw":    {"unit": "kW"},
                "standby_loss_kw": {"unit": "kW"},
                "cycling_loss_kw": {"unit": "kW"},
                "flue_gas_temp_c": {"unit": "degC"},
            },
            "source": "EN 303-5:2012; Obernberger & Thek (2008); Jenkins et al. (1998)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC087 F1b — Biomass Boiler:")
    for plr in [0.15, 0.3, 0.5, 0.7, 0.9, 1.0]:
        r = model.predict({"PLR": plr})
        print(f"  PLR={plr:.2f}: eta={float(r['efficiency']):.3f}, "
              f"Q_flue={float(r['flue_loss_kw']):.2f} kW, "
              f"Q_cycle={float(r['cycling_loss_kw']):.2f} kW, "
              f"T_flue={float(r['flue_gas_temp_c']):.1f} degC")
