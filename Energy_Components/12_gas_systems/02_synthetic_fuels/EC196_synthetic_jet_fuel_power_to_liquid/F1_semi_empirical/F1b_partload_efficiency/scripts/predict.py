"""EC196 — Synthetic Jet Fuel — F1b Part-Load + Thermal — Standard predict interface."""

import json
import numpy as np
from pathlib import Path
from model import FTJetFuelF1b


class ComponentModel:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = Path(__file__).parent.parent / "data" / "parameters.json"
        with open(model_path, "r") as f:
            params = json.load(f)
        self.model = FTJetFuelF1b(params)
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
            "ec_id": "EC196",
            "component": "Synthetic Jet Fuel (Power-to-Liquid)",
            "fidelity": "F1b",
            "sub_fidelity": "partload_efficiency",
            "description": "FT LTFT: part-load PLR correction, alpha(T) ASF, exothermic heat recovery, catalyst deactivation.",
            "inputs": ["n_co_in (mol/s)", "T_set (degC)", "pressure_bar",
                       "plr (opt)", "operating_hours (opt)"],
            "outputs": ["co_conversion", "effective_temperature_C", "alpha_ASF",
                        "selectivity_jet_C8_C16", "jet_fuel_mol_s",
                        "heat_recovery_kW", "energy_efficiency", "deactivation_factor"],
            "references": ["Dry (2002)", "Hillestad (2018)", "Schulz (1999)"],
        }
