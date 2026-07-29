"""EC117 -- BWR -- F1b Load Following -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import BWRF1b


class ComponentModel:
    """Standardized interface for EC117 BWR -- F1b load-following model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BWRF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "power_fraction":          float [0.6-1.0], current power level
                "time_at_power_hours":     float [0-168], time since last change
                "previous_power_fraction": float [0-1.0], power before change
            }

        Returns:
            dict with:
                power_output_mw           : float, electrical output [MW_e]
                xenon_concentration_rel   : float, Xe relative to full-power equilibrium
                available_reactivity_pcm  : float, available reactivity [pcm]
                void_reactivity_pcm       : float, void feedback reactivity [pcm]
                ramp_rate_limit_pct_min   : float, max ramp rate [%/min]
                can_restart               : bool, whether restart to full power possible
        """
        return self._model.predict(
            power_fraction=float(inputs.get("power_fraction", 1.0)),
            time_at_power_hours=float(inputs.get("time_at_power_hours", 0.0)),
            previous_power_fraction=float(inputs.get("previous_power_fraction", 1.0)),
        )

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Boiling Water Reactor (BWR)",
            "ec_id": "EC117",
            "fidelity": "F1b",
            "model": "Load-Following with Xenon Dynamics + Void Feedback",
            "description": (
                f"BWR load-following model with Xe-135/I-135 kinetics and void reactivity feedback. "
                f"P_thermal={m.P_thermal:.0f} MW, eta={m.eta:.2f}. "
                f"Ramp rate limit: {m.ramp_limit:.0f} %/min (slower than PWR due to two-phase flow). "
                f"PLR_min={m.PLR_min:.1f} (void stability floor). "
                f"Negative void coefficient: {m.void_coeff:.0f} pcm/%void."
            ),
            "inputs": {
                "power_fraction":          {"unit": "dimensionless", "range": [0.6, 1.0]},
                "time_at_power_hours":     {"unit": "hours", "range": [0.0, 168.0]},
                "previous_power_fraction": {"unit": "dimensionless", "range": [0.0, 1.0]},
            },
            "outputs": {
                "power_output_mw":          {"unit": "MW_e"},
                "xenon_concentration_rel":  {"unit": "dimensionless"},
                "available_reactivity_pcm": {"unit": "pcm"},
                "void_reactivity_pcm":      {"unit": "pcm"},
                "ramp_rate_limit_pct_min":  {"unit": "%/min"},
                "can_restart":              {"unit": "bool"},
            },
            "source": "Todreas & Kazimi (2012); NEA/CSNI (2011) BWR load-following; Duderstadt & Hamilton (1976)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")

    r = model.predict({
        "power_fraction": 1.0,
        "time_at_power_hours": 48.0,
        "previous_power_fraction": 1.0,
    })
    print("\nFull power, equilibrium:")
    for k, v in r.items():
        print(f"  {k}: {v}")

    r2 = model.predict({
        "power_fraction": 0.75,
        "time_at_power_hours": 10.0,
        "previous_power_fraction": 1.0,
    })
    print("\n75% power, 10h after reduction from 100%:")
    for k, v in r2.items():
        print(f"  {k}: {v}")
