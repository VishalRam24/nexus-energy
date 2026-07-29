"""EC181 — Transmission Line — F1b Thermal Ampacity — Standard predict interface."""

import json
import numpy as np
from pathlib import Path
from model import TransmissionLineF1b


class ComponentModel:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = Path(__file__).parent.parent / "data" / "parameters.json"
        with open(model_path, "r") as f:
            params = json.load(f)
        self.model = TransmissionLineF1b(params)
        self._params = params

    def predict(self, inputs: dict) -> dict:
        """
        inputs keys:
            V_s_pu        : sending-end voltage magnitude [pu]
            delta_s_rad   : sending-end angle [rad] (default 0)
            P_load_pu     : receiving-end active load [pu]
            Q_load_pu     : receiving-end reactive load [pu] (default 0)
            length_km     : optional line length [km]
            T_cond_C      : optional conductor temperature [degC]
            T_amb_C       : optional ambient temperature [degC] (default 25)
        """
        return self.model.compute(
            V_s_pu      = np.asarray(inputs["V_s_pu"],      dtype=float),
            delta_s_rad = np.asarray(inputs.get("delta_s_rad", 0.0), dtype=float),
            P_load_pu   = np.asarray(inputs["P_load_pu"],   dtype=float),
            Q_load_pu   = np.asarray(inputs.get("Q_load_pu", 0.0), dtype=float),
            length_km   = inputs.get("length_km", None),
            T_cond_C    = inputs.get("T_cond_C", None),
            T_amb_C     = inputs.get("T_amb_C", 25.0),
        )

    def get_info(self) -> dict:
        return {
            "ec_id": "EC181",
            "component": "Transmission Line",
            "fidelity": "F1b",
            "sub_fidelity": "thermal_ampacity",
            "description": "Pi-model with R(T) temperature correction, skin-effect, and IEEE 738 thermal ampacity limit.",
            "inputs": ["V_s_pu", "delta_s_rad", "P_load_pu", "Q_load_pu",
                       "length_km (opt)", "T_cond_C (opt)", "T_amb_C (opt)"],
            "outputs": ["V_r_pu", "delta_r_rad", "I_series_pu", "I_series_A",
                        "P_loss_pu", "Q_loss_pu", "efficiency", "voltage_drop_pu",
                        "R_ac_pu_total", "skin_factor", "I_max_A",
                        "ampacity_margin", "derating_factor"],
            "physics": ["R(T) via IEC 60287 alpha coefficient",
                        "Skin effect: R_ac = R_dc * skin_factor",
                        "IEEE 738 ampacity: forced+natural convection + radiation - solar"],
            "references": ["IEEE Std 738-2012", "IEC 60287-1-1 (2006)", "Glover 2012 5th ed."],
        }
