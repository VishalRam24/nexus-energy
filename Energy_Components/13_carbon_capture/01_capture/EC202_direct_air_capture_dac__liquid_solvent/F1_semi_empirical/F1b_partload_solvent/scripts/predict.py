"""EC202 — DAC Liquid Solvent — F1b Part-Load + Solvent Degradation — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import DACLSF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = DACLSF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict liquid-solvent DAC performance at part-load with solvent degradation.

        Parameters
        ----------
        inputs : dict
            air_flow_m3h            : float (m3/h)          default 3.6e6
            T_calciner_degC         : float (degC)           default 900.0
            plr                     : float (0.2-1.0)        default 1.0
            operating_hours         : float (hours)           default 0.0
            injection_pressure_bar  : float (bar)            default 100.0
        """
        air_flow = inputs.get("air_flow_m3h", 3.6e6)
        T_calc   = inputs.get("T_calciner_degC", 900.0)
        plr      = np.asarray(inputs.get("plr", 1.0), dtype=float)
        hours    = inputs.get("operating_hours", 0.0)
        P_inj    = inputs.get("injection_pressure_bar", 100.0)

        return self._model.compute(air_flow, T_calc, plr, hours, P_inj)

    def get_info(self) -> dict:
        return {
            "name": "Direct Air Capture — Liquid Solvent (KOH/Ca(OH)2)",
            "ec_id": "EC202",
            "fidelity": "F1b",
            "description": (
                "KOH/Ca(OH)2 liquid-solvent DAC (Carbon Engineering style). "
                "Models part-load contactor penalty, calciner temperature effects, "
                "KOH capacity degradation due to K2CO3 poisoning (~5%/yr), "
                "and CO2 injection pressure compression energy."
            ),
            "inputs": {
                "air_flow_m3h":           {"unit": "m3/h", "range": [1e5, 1e8]},
                "T_calciner_degC":        {"unit": "degC", "range": [750, 950]},
                "plr":                    {"unit": "dimensionless", "range": [0.2, 1.0]},
                "operating_hours":        {"unit": "hours", "range": [0, 87600]},
                "injection_pressure_bar": {"unit": "bar", "range": [1, 300]},
            },
            "outputs": {
                "capture_rate_kgh":             {"unit": "kg/h"},
                "calciner_duty_gj_ton":         {"unit": "GJ/tCO2"},
                "electrical_kwh_ton":           {"unit": "kWh_e/tCO2"},
                "liquefaction_energy_gj_ton":   {"unit": "GJ/tCO2"},
                "total_specific_energy_gj_ton": {"unit": "GJ/tCO2"},
                "solvent_capacity_pct":         {"unit": "%"},
            },
            "source": "Keith et al. (2018) Joule; McQueen et al. (2021) Progress in Energy",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"plr": 1.0, "operating_hours": 0, "injection_pressure_bar": 100.0})
    print("Design point (PLR=1.0, fresh solvent, P_inj=100 bar):")
    for k, v in r.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")
