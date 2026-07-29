"""EC188 — SMES — F1b Self-Discharge + Cryogenic — Standard predict interface."""

import json
import numpy as np
from pathlib import Path
from model import SMESF1b


class ComponentModel:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = Path(__file__).parent.parent / "data" / "parameters.json"
        with open(model_path, "r") as f:
            params = json.load(f)
        self.model = SMESF1b(params)
        self._params = params

    def predict(self, inputs: dict) -> dict:
        """
        inputs keys:
            SOC           : state of charge [0-1]
            P_request_MW  : power request [MW]
            mode          : "charge" or "discharge" (default "discharge")
            dt_s          : time step [s] (default 1.0)
            T_op_K        : optional operating temperature [K]
        """
        return self.model.compute(
            SOC          = np.asarray(inputs["SOC"], dtype=float),
            P_request_MW = np.asarray(inputs["P_request_MW"], dtype=float),
            mode         = inputs.get("mode", "discharge"),
            dt_s         = inputs.get("dt_s", 1.0),
            T_op_K       = inputs.get("T_op_K", None),
        )

    def get_info(self) -> dict:
        return {
            "ec_id": "EC188",
            "component": "SMES",
            "fidelity": "F1b",
            "sub_fidelity": "self_discharge_cryo",
            "description": "SMES energy model with AC losses (Norris), cryogenic cooling load (Carnot COP), self-discharge time constant.",
            "inputs": ["SOC", "P_request_MW", "mode", "dt_s (opt)", "T_op_K (opt)"],
            "outputs": ["P_delivered_MW", "P_grid_MW", "P_cryo_load_MW", "P_ac_loss_MW",
                        "SOC_new", "E_stored_MJ", "I_coil_A",
                        "eta_instantaneous", "eta_rt_estimate", "self_discharge_tau_h"],
            "physics": ["AC losses: Norris (1970) hysteresis model, P_ac = k*I^n",
                        "Cryo load: Carnot COP with actual efficiency fraction",
                        "Self-discharge: tau = E/(P_cryo + P_ac)"],
            "references": ["Buckles & Hassenzahl (2000)", "Kalsi (2011) Wiley",
                           "Norris (1970) J. Phys. D"],
        }
