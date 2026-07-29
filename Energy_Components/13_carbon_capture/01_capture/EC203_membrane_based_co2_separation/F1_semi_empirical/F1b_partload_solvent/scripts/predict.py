"""EC203 — Membrane CO2 Separation — F1b Part-Load + Aging — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import MembraneF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MembraneF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict membrane CO2 separation performance.

        Parameters
        ----------
        inputs : dict
            co2_feed_fraction       : float (mol/mol)   default 0.12
            feed_flow_mol_s         : float (mol/s)     default 100.0
            capture_rate_target     : float (0-0.90)    default 0.60
            T_feed_degC             : float (degC)      default 40.0
            operating_hours         : float (hours)     default 0.0
            injection_pressure_bar  : float (bar)       default 100.0
        """
        x_CO2   = inputs.get("co2_feed_fraction", 0.12)
        flow    = inputs.get("feed_flow_mol_s", 100.0)
        cr      = np.asarray(inputs.get("capture_rate_target", 0.60), dtype=float)
        T       = inputs.get("T_feed_degC", 40.0)
        hours   = inputs.get("operating_hours", 0.0)
        P_inj   = inputs.get("injection_pressure_bar", 100.0)

        return self._model.compute(x_CO2, flow, cr, T, hours, P_inj)

    def get_info(self) -> dict:
        return {
            "name": "Membrane-Based CO2 Separation",
            "ec_id": "EC203",
            "fidelity": "F1b",
            "description": (
                "Polymeric membrane CO2 separation with physical aging (permeance decline ~6%/yr), "
                "Arrhenius temperature correction, capture-rate-dependent SEC, "
                "and multi-stage CO2 compression energy for injection."
            ),
            "inputs": {
                "co2_feed_fraction":      {"unit": "mol/mol", "range": [0.04, 0.20]},
                "feed_flow_mol_s":        {"unit": "mol/s",   "range": [1, 1000]},
                "capture_rate_target":    {"unit": "dimensionless", "range": [0.30, 0.90]},
                "T_feed_degC":            {"unit": "degC",    "range": [20, 80]},
                "operating_hours":        {"unit": "hours",   "range": [0, 87600]},
                "injection_pressure_bar": {"unit": "bar",     "range": [10, 200]},
            },
            "outputs": {
                "co2_captured_kg_h":    {"unit": "kg/h"},
                "sec_kwh_ton":          {"unit": "kWh_e/tCO2"},
                "compression_kwh_ton":  {"unit": "kWh_e/tCO2"},
                "total_energy_kwh_ton": {"unit": "kWh_e/tCO2"},
                "permeance_pct":        {"unit": "%"},
            },
            "source": "Merkel et al. (2010) J. Membrane Sci.; Baker & Low (2014) Macromolecules",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"capture_rate_target": 0.60, "operating_hours": 0})
    print("Design point (CR=0.60, fresh membrane, T=40C):")
    for k, v in r.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")
