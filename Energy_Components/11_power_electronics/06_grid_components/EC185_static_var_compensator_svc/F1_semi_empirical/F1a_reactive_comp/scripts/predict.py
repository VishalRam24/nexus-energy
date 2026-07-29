"""EC185 — SVC — F1a Reactive Compensation — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import SVCModel


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SVCModel(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            Q_demand_MVAR : float or array [MVAR]  requested reactive output
            V_pu          : float (optional)       terminal voltage [pu]
        returns:
            Q_out_MVAR    : [MVAR]  actual output (clamped to [Q_min, Q_max])
            Q_limited     : [bool]  True if demand was clamped
            P_loss_MW     : [MW]    active power losses
            operating_mode: str     "capacitive"/"inductive"/"standby"
            utilization   : [—]     fractional use of rated capacity
        """
        return self._model.compute(
            Q_demand_MVAR=inputs["Q_demand_MVAR"],
            V_pu=inputs.get("V_pu", 1.0),
        )

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Static VAR Compensator (SVC, TCR+TSC)",
            "ec_id": "EC185",
            "fidelity": "F1a",
            "description": "Q_out=clamp(Q_demand, Q_min, Q_max); P_loss=loss_factor*|Q_out|",
            "inputs": {
                "Q_demand_MVAR": {"unit": "MVAR", "range": [-50.0, 100.0]},
                "V_pu": {"unit": "pu", "range": [0.85, 1.15], "optional": True},
            },
            "outputs": {
                "Q_out_MVAR": {"unit": "MVAR"},
                "Q_limited": {"unit": "bool"},
                "P_loss_MW": {"unit": "MW"},
                "operating_mode": {"unit": "enum"},
                "utilization": {"unit": "dimensionless"},
            },
            "params": {
                "Q_max": f"+{u['Q_max_MVAR']['value']} MVAR (capacitive)",
                "Q_min": f"{u['Q_min_MVAR']['value']} MVAR (inductive)",
                "loss_factor": f"{u['loss_factor']['value']*100:.1f}%",
                "response_time": f"{u['response_time_ms']['value']} ms",
            },
            "source": "Hingorani & Gyugyi (2000), Understanding FACTS, IEEE Press",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for Q_req in [-60, -30, 0, 50, 80, 120]:
        r = model.predict({"Q_demand_MVAR": float(Q_req)})
        print(f"Q_req={Q_req:>5} MVAR → Q_out={float(r['Q_out_MVAR']):>7.1f}  "
              f"mode={r['operating_mode']:<12}  P_loss={float(r['P_loss_MW']):.3f} MW")
