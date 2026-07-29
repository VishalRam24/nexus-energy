"""EC175 — Induction Motor/Generator — F1b Thermal — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import InductionMotorF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = InductionMotorF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            load_fraction       : float or array [0.05, 1.2]
            winding_temperature : float or array [degC] (default 75)
            ambient_temperature : float or array [degC] (default 25)
        returns:
            efficiency          : float or array
            input_power_kw      : float or array [kW]
            output_power_kw     : float or array [kW]
            losses_kw           : float or array [kW]
            current_A           : float or array [A]
            derating_factor     : float or array
            slip                : float or array
        """
        plr = np.asarray(inputs["load_fraction"], dtype=float)
        T_w = inputs.get("winding_temperature", 75.0)
        T_a = inputs.get("ambient_temperature", 25.0)
        return {
            "efficiency": self._model.efficiency(plr, T_w),
            "input_power_kw": self._model.input_power(plr, T_w, T_a),
            "output_power_kw": self._model.output_power(plr, T_a),
            "losses_kw": self._model.losses(plr, T_w, T_a),
            "current_A": self._model.current(plr, T_w, T_a),
            "derating_factor": self._model.derating_factor(T_a),
            "slip": self._model.slip(plr),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Induction Motor/Generator (Thermal)",
            "ec_id": "EC175",
            "fidelity": "F1b",
            "description": (
                "Temperature-dependent efficiency: R(T) = R_ref*(1+alpha_Cu*(T-T_ref)); "
                "IEC 60034-1 ambient derating above 40C"
            ),
            "inputs": {
                "load_fraction": {"unit": "dimensionless", "range": [0.05, 1.2]},
                "winding_temperature": {"unit": "degC", "range": [20, 180], "default": 75},
                "ambient_temperature": {"unit": "degC", "range": [-20, 60], "default": 25},
            },
            "outputs": {
                "efficiency": {"unit": "dimensionless"},
                "input_power_kw": {"unit": "kW"},
                "output_power_kw": {"unit": "kW"},
                "losses_kw": {"unit": "kW"},
                "current_A": {"unit": "A"},
                "derating_factor": {"unit": "dimensionless"},
                "slip": {"unit": "dimensionless"},
            },
            "params": {
                "P_rated": f"{u['rated_power_kw']['value']} kW",
                "R_ref": f"{u['R_ref']['value']} ohm at {u['T_ref']['value']}C",
                "alpha_Cu": f"{u['alpha_Cu']['value']} 1/K",
            },
            "source": "IEC 60034-30-1; IEC 60034-1; Boldea & Nasar (2010)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for T_w in [25, 75, 120, 155]:
        r = model.predict({"load_fraction": 1.0, "winding_temperature": T_w})
        print(
            f"T_winding={T_w}C: eta={float(r['efficiency']):.4f}  "
            f"P_in={float(r['input_power_kw']):.2f} kW  "
            f"loss={float(r['losses_kw']):.3f} kW  "
            f"I={float(r['current_A']):.2f} A"
        )
