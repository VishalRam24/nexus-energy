"""EC197 — DME Synthesis Reactor — F1b Part-Load + Thermal — Standard predict interface."""

import json
import numpy as np
from pathlib import Path
from model import DMEReactorF1b


class ComponentModel:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = Path(__file__).parent.parent / "data" / "parameters.json"
        with open(model_path, "r") as f:
            params = json.load(f)
        self.model = DMEReactorF1b(params)
        self._params = params

    def predict(self, inputs: dict) -> dict:
        return self.model.compute(
            n_co_in         = np.asarray(inputs.get("n_co_in", self.model.n_CO_in), dtype=float),
            T_set           = np.asarray(inputs["T_set"], dtype=float),
            pressure_bar    = np.asarray(inputs["pressure_bar"], dtype=float),
            plr             = np.asarray(inputs.get("plr", 1.0), dtype=float),
            operating_hours = inputs.get("operating_hours", 0.0),
        )

    def get_info(self) -> dict:
        return {
            "ec_id": "EC197",
            "component": "DME Synthesis Reactor",
            "fidelity": "F1b",
            "sub_fidelity": "partload_efficiency",
            "description": "DME reactor with PLR correction, bed T drop, coking deactivation, heat recovery, MeOH slip.",
            "inputs": ["n_co_in", "T_set", "pressure_bar", "plr (opt)", "operating_hours (opt)"],
            "outputs": ["co_conversion", "effective_temperature_C", "selectivity_dme",
                        "dme_production_mol_s", "meoh_slip_mol_s",
                        "heat_recovery_kW", "energy_efficiency", "deactivation_factor"],
            "references": ["Ereña (2005)", "García-Trenco (2018)", "Naik (2011)"],
        }
