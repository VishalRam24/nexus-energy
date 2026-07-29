"""EC182 — Distribution Line — F1b Thermal Ampacity — Standard predict interface."""

import json
import numpy as np
from pathlib import Path
from model import DistributionLineF1b


class ComponentModel:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = Path(__file__).parent.parent / "data" / "parameters.json"
        with open(model_path, "r") as f:
            params = json.load(f)
        self.model = DistributionLineF1b(params)
        self._params = params

    def predict(self, inputs: dict) -> dict:
        """
        inputs keys:
            V_s_kV      : sending-end line-to-line voltage [kV]
            P_load_kW   : active load [kW]
            Q_load_kVAR : reactive load [kVAR] (default 0)
            length_km   : optional feeder length [km]
            T_cond_C    : optional conductor temperature [degC]
            T_amb_C     : optional ambient temperature [degC] (default 25)
        """
        return self.model.compute(
            V_s_kV      = np.asarray(inputs["V_s_kV"], dtype=float),
            P_load_kW   = np.asarray(inputs["P_load_kW"], dtype=float),
            Q_load_kVAR = np.asarray(inputs.get("Q_load_kVAR", 0.0), dtype=float),
            length_km   = inputs.get("length_km", None),
            T_cond_C    = inputs.get("T_cond_C", None),
            T_amb_C     = inputs.get("T_amb_C", 25.0),
        )

    def get_info(self) -> dict:
        return {
            "ec_id": "EC182",
            "component": "Distribution Line",
            "fidelity": "F1b",
            "sub_fidelity": "thermal_ampacity",
            "description": "R+jX series model with R(T) temperature correction, skin effect, and IEC 60287 / IEEE 738 thermal ampacity.",
            "inputs": ["V_s_kV", "P_load_kW", "Q_load_kVAR (opt)",
                       "length_km (opt)", "T_cond_C (opt)", "T_amb_C (opt)"],
            "outputs": ["V_r_kV", "I_line_A", "P_loss_kW", "Q_loss_kVAR",
                        "P_s_kW", "efficiency", "voltage_drop_kV", "voltage_drop_pct",
                        "power_factor_load", "R_ac_ohm_km", "skin_factor",
                        "I_max_A", "ampacity_margin", "congestion_factor", "derating_factor"],
            "physics": ["R(T) via IEC 60287 alpha_R",
                        "Skin effect: R_ac = skin_factor * R_dc",
                        "Overhead ampacity: IEEE 738 convection+radiation-solar",
                        "Underground ampacity: IEC 60287 thermal resistance method"],
            "references": ["Kersting (2012)", "IEC 60287-1-1 (2006)", "IEEE Std 738-2012"],
        }
