"""EC120 -- FBR -- F1b Load-Following -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import FBRF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = FBRF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            power_fraction          : float [0.25-1.0]
            time_at_power_hours     : float [0-168]
            previous_power_fraction : float [0-1.0]
        returns:
            power_output_mw             : electrical output [MW_e]
            xenon_concentration_rel     : Xe (negligible impact in fast spectrum)
            sodium_void_reactivity_pcm  : sodium void feedback [pcm]
            doppler_reactivity_pcm      : Doppler feedback [pcm]
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
            "name":        "Fast Breeder Reactor (FBR)",
            "ec_id":       "EC120",
            "fidelity":    "F1b",
            "model":       "Load-Following with Void/Doppler Feedback",
            "description": (
                f"Sodium-cooled FBR load-following: Xe poisoning negligible in fast spectrum "
                f"(σ_Xe={m.sigma_Xe:.2e} cm2, Xe worth ~{abs(m.xe_react_coeff):.0f} pcm); "
                f"sodium void coeff +{m.void_coeff:.0f} pcm/%void (positive in large core); "
                f"Doppler coeff {m.doppler_coeff:.1f} pcm/K (negative, stabilizing). "
                f"P_thermal={m.P_thermal:.0f} MW, eta={m.eta:.2f}. "
                f"Ramp rate {m.ramp_limit:.0f}%/min (limited by Na pool thermal gradients)."
            ),
            "inputs": {
                "power_fraction":          {"unit": "dimensionless", "range": [0.25, 1.0]},
                "time_at_power_hours":     {"unit": "hours",         "range": [0.0, 168.0]},
                "previous_power_fraction": {"unit": "dimensionless", "range": [0.0, 1.0]},
            },
            "outputs": {
                "power_output_mw":            {"unit": "MW_e"},
                "xenon_concentration_rel":    {"unit": "dimensionless (negligible)"},
                "sodium_void_reactivity_pcm": {"unit": "pcm"},
                "doppler_reactivity_pcm":     {"unit": "pcm"},
                "available_reactivity_pcm":   {"unit": "pcm"},
                "ramp_rate_limit_pct_min":    {"unit": "%/min"},
                "can_restart":                {"unit": "bool"},
            },
            "source": (
                "Guidez & Prele (2017) Sodium Cooled Fast Reactors EDP; "
                "Koch (2008) EBR-II; IAEA-TECDOC-1689 (2012)"
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
