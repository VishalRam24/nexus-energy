"""EC116 -- PWR -- F1b Load Following -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import PWRF1b


class ComponentModel:
    """Standardized interface for EC116 PWR -- F1b load-following model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PWRF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "power_fraction":          float [0.3-1.0], current power level
                "time_at_power_hours":     float [0-168], time since last change
                "previous_power_fraction": float [0-1.0], power before change
            }

        Returns:
            dict with:
                power_output_mw           : float, electrical output [MW_e]
                xenon_concentration_rel   : float, Xe relative to eq at full power
                available_reactivity_pcm  : float, available reactivity [pcm]
                ramp_rate_limit_pct_min   : float, max ramp rate [%/min]
                can_restart               : bool, whether restart is possible
        """
        return self._model.predict(
            power_fraction=float(inputs.get("power_fraction", 1.0)),
            time_at_power_hours=float(inputs.get("time_at_power_hours", 0.0)),
            previous_power_fraction=float(inputs.get("previous_power_fraction", 1.0)),
        )

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Pressurized Water Reactor (PWR)",
            "ec_id": "EC116",
            "fidelity": "F1b",
            "model": "Load-Following with Xenon Dynamics",
            "description": (
                f"PWR load-following model with Xe-135/I-135 kinetics. "
                f"P_thermal={m.P_thermal:.0f} MW, eta={m.eta:.2f}. "
                f"Ramp rate limit: {m.ramp_limit:.0f} %/min. "
                f"Xenon peak ~8-12h after power reduction. "
                f"Tracks available reactivity margin for restart capability."
            ),
            "inputs": {
                "power_fraction":          {"unit": "dimensionless", "range": [0.3, 1.0]},
                "time_at_power_hours":     {"unit": "hours", "range": [0.0, 168.0]},
                "previous_power_fraction": {"unit": "dimensionless", "range": [0.0, 1.0]},
            },
            "outputs": {
                "power_output_mw":          {"unit": "MW_e"},
                "xenon_concentration_rel":  {"unit": "dimensionless"},
                "available_reactivity_pcm": {"unit": "pcm"},
                "ramp_rate_limit_pct_min":  {"unit": "%/min"},
                "can_restart":              {"unit": "bool"},
            },
            "source": "Todreas & Kazimi (2012); Stacey (2007); Duderstadt & Hamilton (1976)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")

    # Steady state at full power
    r = model.predict({
        "power_fraction": 1.0,
        "time_at_power_hours": 48.0,
        "previous_power_fraction": 1.0,
    })
    print(f"\nFull power, equilibrium:")
    for k, v in r.items():
        print(f"  {k}: {v}")

    # After power reduction to 50%
    r2 = model.predict({
        "power_fraction": 0.5,
        "time_at_power_hours": 10.0,
        "previous_power_fraction": 1.0,
    })
    print(f"\n50% power, 10h after reduction from 100%:")
    for k, v in r2.items():
        print(f"  {k}: {v}")
