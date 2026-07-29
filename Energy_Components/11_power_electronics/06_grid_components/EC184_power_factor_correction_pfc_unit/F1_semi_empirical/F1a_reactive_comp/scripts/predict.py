"""EC184 — PFC Unit — F1a Reactive Compensation — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import PFCUnitModel


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PFCUnitModel(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            P_kW                 : float or array [kW]  active load
            pf_initial           : float or array [—]   initial power factor (lagging)
            pf_target            : float (optional)     target pf (default 0.95)
            Q_comp_override_kVAR : float (optional)     fixed Q_comp, bypasses pf_target calc
        returns:
            Q_load_kVAR       : [kVAR]  original reactive load
            Q_required_kVAR   : [kVAR]  compensation needed to reach pf_target
            Q_compensated_kVAR: [kVAR]  actual compensation (capped by Q_rated)
            Q_residual_kVAR   : [kVAR]  remaining reactive demand after compensation
            pf_achieved       : [—]     achieved power factor
            P_loss_kW         : [kW]    capacitor bank losses
            bank_utilization  : [—]     Q_comp / Q_rated
        """
        return self._model.compute(
            P_kW=inputs["P_kW"],
            pf_initial=inputs["pf_initial"],
            pf_target=inputs.get("pf_target", None),
            Q_comp_override_kVAR=inputs.get("Q_comp_override_kVAR", None),
        )

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Power Factor Correction Unit (Capacitor Bank)",
            "ec_id": "EC184",
            "fidelity": "F1a",
            "description": "Q_comp=P*(tan(phi1)-tan(phi2)); P_loss=loss_factor*Q_comp",
            "inputs": {
                "P_kW": {"unit": "kW", "range": [0.0, 10000.0]},
                "pf_initial": {"unit": "dimensionless", "range": [0.5, 0.99]},
                "pf_target": {"unit": "dimensionless", "range": [0.8, 1.0], "optional": True},
                "Q_comp_override_kVAR": {"unit": "kVAR", "optional": True},
            },
            "outputs": {
                "Q_load_kVAR": {"unit": "kVAR"},
                "Q_required_kVAR": {"unit": "kVAR"},
                "Q_compensated_kVAR": {"unit": "kVAR"},
                "Q_residual_kVAR": {"unit": "kVAR"},
                "pf_achieved": {"unit": "dimensionless"},
                "P_loss_kW": {"unit": "kW"},
                "bank_utilization": {"unit": "dimensionless"},
            },
            "params": {
                "Q_rated": f"{u['Q_rated_kVAR']['value']} kVAR",
                "pf_target_default": u["pf_target"]["value"],
                "loss_factor": u["loss_factor"]["value"],
            },
            "source": "Acha et al. (2004), FACTS: Modelling and Simulation in Power Networks; IEEE Std 1036-2010",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"P_kW": 2000.0, "pf_initial": 0.75, "pf_target": 0.95})
    print(f"Q_required={float(r['Q_required_kVAR']):.1f} kVAR  "
          f"Q_comp={float(r['Q_compensated_kVAR']):.1f} kVAR  "
          f"pf_achieved={float(r['pf_achieved']):.4f}  "
          f"P_loss={float(r['P_loss_kW']):.3f} kW")
