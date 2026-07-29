"""EC199 — Pre-Combustion Capture — F1a Capture Rate + Energy — Standard predict interface."""

import json
import numpy as np
from pathlib import Path
from model import PreCombustionCaptureF1a


class ComponentModel:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = Path(__file__).parent.parent / "data" / "parameters.json"
        with open(model_path, "r") as f:
            params = json.load(f)
        self.model = PreCombustionCaptureF1a(params)
        self._params = params

    def predict(self, inputs: dict) -> dict:
        return self.model.compute(
            syngas_flow_mol_s = np.asarray(inputs.get("syngas_flow_mol_s", 100.0), dtype=float),
            co_fraction       = np.asarray(inputs["co_fraction"], dtype=float),
            h2_fraction       = np.asarray(inputs.get("h2_fraction", 0.35), dtype=float),
            T_WGS_C           = np.asarray(inputs.get("T_WGS_C", 250.0), dtype=float),
            P_bar             = np.asarray(inputs["P_bar"], dtype=float),
            steam_co_ratio    = inputs.get("steam_co_ratio", 3.0),
        )

    def get_info(self) -> dict:
        return {
            "ec_id": "EC199",
            "component": "Pre-Combustion Capture (WGS + Separation)",
            "fidelity": "F1a",
            "sub_fidelity": "capture_rate_energy",
            "description": "WGS CO conversion + Selexol/Rectisol separation. CO2 capture rate and energy penalty.",
            "inputs": ["syngas_flow_mol_s", "co_fraction", "h2_fraction",
                       "T_WGS_C", "P_bar", "steam_co_ratio (opt)"],
            "outputs": ["wgs_conversion", "capture_rate", "co2_captured_kg_s",
                        "h2_yield_mol_s", "total_energy_GJ_tCO2", "wgs_heat_kW"],
            "references": ["IEAGHG (2014) 2012/8", "Kunze & Spliethoff (2012)",
                           "DOE/NETL-2010/1397"],
        }
