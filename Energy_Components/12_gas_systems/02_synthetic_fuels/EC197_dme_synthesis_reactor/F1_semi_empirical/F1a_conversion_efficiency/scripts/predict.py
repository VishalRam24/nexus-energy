"""EC197 — DME Synthesis Reactor — F1a Conversion Efficiency — Standard predict interface."""

import json
import numpy as np
from pathlib import Path
from model import DMEReactorF1a


class ComponentModel:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = Path(__file__).parent.parent / "data" / "parameters.json"
        with open(model_path, "r") as f:
            params = json.load(f)
        self.model = DMEReactorF1a(params)
        self._params = params

    def predict(self, inputs: dict) -> dict:
        return self.model.compute(
            n_co_in      = np.asarray(inputs.get("n_co_in", self.model.n_CO_in), dtype=float),
            temperature_C= np.asarray(inputs["temperature_C"], dtype=float),
            pressure_bar = np.asarray(inputs["pressure_bar"], dtype=float),
        )

    def get_info(self) -> dict:
        return {
            "ec_id": "EC197",
            "component": "DME Synthesis Reactor",
            "fidelity": "F1a",
            "sub_fidelity": "conversion_efficiency",
            "description": "Single-step DME synthesis: CO conversion, DME selectivity, yield and energy efficiency.",
            "inputs": ["n_co_in (mol/s)", "temperature_C", "pressure_bar"],
            "outputs": ["co_conversion", "selectivity_dme", "dme_production_mol_s",
                        "energy_efficiency", "heat_released_kW"],
            "references": ["Ereña et al. (2005)", "García-Trenco et al. (2018)", "Naik et al. (2011)"],
        }
