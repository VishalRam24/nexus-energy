"""EC187 — HVDC Converter Station — F1b Full Link — Standard predict interface."""

import json
import numpy as np
from pathlib import Path
from model import HVDCLinkF1b


class ComponentModel:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = Path(__file__).parent.parent / "data" / "parameters.json"
        with open(model_path, "r") as f:
            params = json.load(f)
        self.model = HVDCLinkF1b(params)
        self._params = params

    def predict(self, inputs: dict) -> dict:
        """
        inputs keys:
            P_transfer_MW : active power at rectifier DC output [MW]
            T_line_C      : optional DC line temperature [degC] (default 20)
        """
        return self.model.compute(
            P_transfer_MW = np.asarray(inputs["P_transfer_MW"], dtype=float),
            T_line_C      = inputs.get("T_line_C", 20.0),
        )

    def get_info(self) -> dict:
        return {
            "ec_id": "EC187",
            "component": "HVDC Converter Station",
            "fidelity": "F1b",
            "sub_fidelity": "station_line_losses",
            "description": "Full point-to-point HVDC link: rectifier + DC line (R(T)) + inverter + LCC reactive demand.",
            "inputs": ["P_transfer_MW", "T_line_C (opt)"],
            "outputs": ["P_AC_in_MW", "P_AC_out_MW", "P_loss_total_MW",
                        "P_loss_rect_MW", "P_loss_line_MW", "P_loss_inv_MW",
                        "I_dc_kA", "R_line_ohm", "link_efficiency",
                        "Q_reactive_demand_MVAR"],
            "physics": ["Station losses: P_no_load + loss_factor * P",
                        "DC line: I^2*R with temperature-corrected R",
                        "LCC reactive demand: Q = Q_factor * P (Cigre TB 388)"],
            "references": ["Cigre TB 492 (2012)", "Cigre TB 388 (2009)",
                           "Kundur (1994) Ch.8", "ABB (2019) HVDC Technical Overview"],
        }
