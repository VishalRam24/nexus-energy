"""EC119 -- MSR -- F1b Load-Following -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import MSRF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MSRF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            power_fraction          : float [0.2-1.0]
            time_at_power_hours     : float [0-168]
            previous_power_fraction : float [0-1.0]
        returns:
            power_output_mw             : electrical output [MW_e]
            xenon_concentration_rel     : Xe relative to full-power equilibrium
            temperature_reactivity_pcm  : fuel salt temp feedback [pcm]
            available_reactivity_pcm    : available reactivity [pcm]
            ramp_rate_limit_pct_min     : max ramp rate [%/min]
            can_restart                 : bool
        """
        return self._model.predict(
            power_fraction=float(inputs.get("power_fraction", 1.0)),
            time_at_power_hours=float(inputs.get("time_at_power_hours", 0.0)),
            previous_power_fraction=float(inputs.get("previous_power_fraction", 1.0)),
        )

    def get_info(self) -> dict:
        m = self._model
        return {
            "name":        "Molten Salt Reactor (MSR)",
            "ec_id":       "EC119",
            "fidelity":    "F1b",
            "model":       "Load-Following with Xe Dynamics + Online Stripping",
            "description": (
                f"MSR load-following: Xe-135/I-135 kinetics modified by online Xe stripping "
                f"(strip efficiency {m.xe_strip_eff:.0%}); "
                f"strongly negative fuel salt temperature coefficient ({m.fuel_temp_coeff:.1f} pcm/K); "
                f"P_thermal={m.P_thermal:.0f} MW, eta={m.eta:.2f}. "
                f"Ramp rate {m.ramp_limit:.0f}%/min; PLR range [{m.PLR_min:.1f}, 1.0]. "
                f"Xenon deadtime effectively eliminated by online Xe removal."
            ),
            "inputs": {
                "power_fraction":          {"unit": "dimensionless", "range": [0.2, 1.0]},
                "time_at_power_hours":     {"unit": "hours",         "range": [0.0, 168.0]},
                "previous_power_fraction": {"unit": "dimensionless", "range": [0.0, 1.0]},
            },
            "outputs": {
                "power_output_mw":            {"unit": "MW_e"},
                "xenon_concentration_rel":    {"unit": "dimensionless"},
                "temperature_reactivity_pcm": {"unit": "pcm"},
                "available_reactivity_pcm":   {"unit": "pcm"},
                "ramp_rate_limit_pct_min":    {"unit": "%/min"},
                "can_restart":                {"unit": "bool"},
            },
            "source": (
                "Haubenreich & Engel (1970) ORNL-4394; MacPherson (1985) Nucl. Sci. Eng. 90:374; "
                "Delpech et al. (2009) J. Fluorine Chem. 130:11; NEA (2021) MSR Handbook"
            ),
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")
    r = model.predict({"power_fraction": 1.0, "time_at_power_hours": 48.0,
                       "previous_power_fraction": 1.0})
    print(f"\nFull power (equilibrium):")
    for k, v in r.items():
        print(f"  {k}: {v}")
    r2 = model.predict({"power_fraction": 0.5, "time_at_power_hours": 10.0,
                        "previous_power_fraction": 1.0})
    print(f"\n50% power, 10h after reduction:")
    for k, v in r2.items():
        print(f"  {k}: {v}")
