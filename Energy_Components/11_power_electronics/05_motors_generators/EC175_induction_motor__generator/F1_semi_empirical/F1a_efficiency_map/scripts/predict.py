"""EC175 — Induction Motor/Generator — F1a Efficiency Map — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import InductionMotorF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = InductionMotorF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            load_fraction : float or array [0.05, 1.2]  (PLR, dimensionless)
            speed_rpm     : float or array [rpm] (optional, uses slip model if omitted)
        returns:
            efficiency         : float or array (dimensionless)
            input_power_kw     : float or array [kW]
            output_power_kw    : float or array [kW]
            losses_kw          : float or array [kW]
            slip               : float or array (dimensionless)
        """
        plr = np.asarray(inputs["load_fraction"], dtype=float)
        # speed_rpm input is accepted but slip is model-derived (consistent interface)
        return {
            "efficiency": self._model.efficiency(plr),
            "input_power_kw": self._model.input_power(plr),
            "output_power_kw": self._model.output_power(plr),
            "losses_kw": self._model.losses(plr),
            "slip": self._model.slip(plr),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Induction Motor/Generator",
            "ec_id": "EC175",
            "fidelity": "F1a",
            "description": (
                "eta(PLR) = eta_rated * (a0+a1+a2)/(a0+a1*PLR+a2*PLR^2); "
                "IEC 60034-30-1 efficiency class IE3"
            ),
            "inputs": {
                "load_fraction": {"unit": "dimensionless", "range": [0.05, 1.2]},
                "speed_rpm": {"unit": "rpm", "range": [0.0, 1500.0], "optional": True},
            },
            "outputs": {
                "efficiency": {"unit": "dimensionless"},
                "input_power_kw": {"unit": "kW"},
                "output_power_kw": {"unit": "kW"},
                "losses_kw": {"unit": "kW"},
                "slip": {"unit": "dimensionless"},
            },
            "params": {
                "P_rated": f"{u['rated_power']['value']} kW",
                "eta_rated": u["eta_rated"]["value"],
                "IE_class": "IE3",
                "poles": u["poles"]["value"],
                "frequency": f"{u['frequency']['value']} Hz",
            },
            "source": "IEC 60034-30-1; Boldea & Nasar (2010)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for plr in [0.25, 0.50, 0.75, 1.00, 1.10]:
        r = model.predict({"load_fraction": plr})
        print(
            f"PLR={plr:.2f}: eta={float(r['efficiency']):.4f}  "
            f"P_in={float(r['input_power_kw']):.2f} kW  "
            f"P_out={float(r['output_power_kw']):.2f} kW  "
            f"loss={float(r['losses_kw']):.3f} kW  "
            f"slip={float(r['slip']):.4f}"
        )
