"""EC180 — DFIG — F1a Efficiency Map — Standardized Predict Interface"""
import json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import DFIGF1a


class ComponentModel:
    component_id = "EC180"

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = DFIGF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Inputs:
            load_fraction : float or array [0.05, 1.2]
            slip          : float or array [-0.30, 0.30], default 0.0
        Returns:
            efficiency, output_power_w, input_power_w, losses_w, rotor_speed_rpm
        """
        plr = np.asarray(inputs["load_fraction"], dtype=float)
        slip = float(inputs.get("slip", 0.0))

        eta = self._model.efficiency(plr, slip)
        P_out = self._model.output_power(plr)
        P_in = self._model.input_power(plr, slip)
        losses = P_in - P_out
        rpm = self._model.rotor_speed_rpm(slip)

        return {
            "efficiency": eta,
            "output_power_w": P_out,
            "input_power_w": P_in,
            "losses_w": losses,
            "rotor_speed_rpm": float(rpm),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Doubly-Fed Induction Generator (DFIG) Efficiency Map",
            "ec_id": self.component_id,
            "fidelity": "F1a",
            "description": (
                "Simple efficiency map: eta_rated=0.95 with part-load and slip corrections. "
                "Variable speed ±30% around synchronous speed. For wind turbine applications."
            ),
            "inputs": {
                "load_fraction": {"unit": "dimensionless", "range": [0.05, 1.2]},
                "slip": {"unit": "dimensionless", "range": [-0.30, 0.30], "default": 0.0},
            },
            "outputs": {
                "efficiency": {"unit": "dimensionless"},
                "output_power_w": {"unit": "W"},
                "input_power_w": {"unit": "W"},
                "losses_w": {"unit": "W"},
                "rotor_speed_rpm": {"unit": "rpm"},
            },
            "params": {
                "P_rated": f"{u['P_rated_MW']['value']} MW",
                "eta_rated": str(u["eta_rated"]["value"]),
                "slip_range": f"[{u['slip_min']['value']}, {u['slip_max']['value']}]",
            },
            "source": "Muller et al. (2002); IEC 61400-21",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for plr in [0.25, 0.5, 0.75, 1.0]:
        for slip in [-0.25, 0.0, 0.25]:
            r = model.predict({"load_fraction": plr, "slip": slip})
            print(f"PLR={plr:.2f} slip={slip:+.2f}: eta={float(r['efficiency']):.4f}  "
                  f"P_out={float(r['output_power_w'])/1e6:.3f} MW  rpm={r['rotor_speed_rpm']:.0f}")
