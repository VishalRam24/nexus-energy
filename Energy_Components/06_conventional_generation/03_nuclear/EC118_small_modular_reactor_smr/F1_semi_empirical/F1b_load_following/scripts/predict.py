"""EC118 -- SMR -- F1b Load-Following + Thermal Inertia -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import SMRF1b


class ComponentModel:
    """Standardized interface for EC118 SMR -- F1b load-following + thermal inertia model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SMRF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "power_fraction":                  float [0.2-1.0], current power level
                "time_at_power_hours":             float [0-168], time since last Xe-relevant change
                "previous_power_fraction":         float [0-1.0], power before change
                "time_since_ramp_start_minutes":   float (optional), time since ramp start [min]
            }

        Returns dict with:
            power_output_mw              : float, electrical output [MW_e]
            thermal_power_mw             : float, reactor thermal power [MW_th]
            coolant_outlet_temp_c        : float, hot-leg temperature [degC]
            xenon_concentration_rel      : float, Xe relative to full-power equilibrium
            available_reactivity_pcm     : float, available reactivity [pcm]
            ramp_rate_limit_pct_min      : float, max ramp rate [%/min]
            can_restart                  : bool, restart capability
            thermal_lag_power_fraction   : float, effective power fraction after lag
        """
        return self._model.predict(
            power_fraction=float(inputs.get("power_fraction", 1.0)),
            time_at_power_hours=float(inputs.get("time_at_power_hours", 0.0)),
            previous_power_fraction=float(inputs.get("previous_power_fraction", 1.0)),
            time_since_ramp_start_minutes=inputs.get("time_since_ramp_start_minutes"),
        )

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Small Modular Reactor (SMR)",
            "ec_id": "EC118",
            "fidelity": "F1b",
            "model": "Deep Load-Following (20-100%) + Xenon Dynamics + Thermal Inertia",
            "description": (
                f"Integral PWR-type SMR. P_thermal={m.P_thermal:.0f} MW, eta={m.eta:.2f}. "
                f"PLR_min={m.PLR_min:.1f} (designed for deep load-following). "
                f"Ramp rate: {m.ramp_limit:.0f} %/min. "
                f"Thermal time constant: {m.tau_min:.0f} min (faster than large PWR). "
                f"First-order thermal lag on coolant temperature during ramps."
            ),
            "inputs": {
                "power_fraction":                {"unit": "dimensionless", "range": [0.2, 1.0]},
                "time_at_power_hours":           {"unit": "hours", "range": [0.0, 168.0]},
                "previous_power_fraction":       {"unit": "dimensionless", "range": [0.0, 1.0]},
                "time_since_ramp_start_minutes": {"unit": "minutes", "range": [0.0, 60.0], "optional": True},
            },
            "outputs": {
                "power_output_mw":            {"unit": "MW_e"},
                "thermal_power_mw":           {"unit": "MW_th"},
                "coolant_outlet_temp_c":      {"unit": "degC"},
                "xenon_concentration_rel":    {"unit": "dimensionless"},
                "available_reactivity_pcm":   {"unit": "pcm"},
                "ramp_rate_limit_pct_min":    {"unit": "%/min"},
                "can_restart":                {"unit": "bool"},
                "thermal_lag_power_fraction": {"unit": "dimensionless"},
            },
            "source": "IAEA SMR Booklet (2022); NuScale VOYGR DCA (2020); Todreas & Kazimi (2012); Ingersoll & Carelli (2020)",
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
    print("\nFull power, equilibrium, steady-state:")
    for k, v in r.items():
        print(f"  {k}: {v}")

    r2 = model.predict({
        "power_fraction": 0.3,
        "time_at_power_hours": 10.0,
        "previous_power_fraction": 1.0,
        "time_since_ramp_start_minutes": 5.0,
    })
    print("\n30% power, 10h Xe transient, 5 min into ramp (thermal lag active):")
    for k, v in r2.items():
        print(f"  {k}: {v}")
