"""EC186 — STATCOM — F1a Reactive Compensation — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import STATCOMModel


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = STATCOMModel(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            Q_demand_MVAR : float or array [MVAR]  requested reactive output (+ = capacitive)
            V_pu          : float (optional)       terminal voltage [pu]
        returns:
            Q_out_MVAR      : [MVAR]  actual output (clamped)
            Q_limited       : [bool]  True if clamped
            P_loss_MW       : [MW]    variable switching losses
            P_standby_MW    : [MW]    fixed standby/cooling losses
            P_total_loss_MW : [MW]    total losses
            operating_mode  : str     "capacitive"/"inductive"/"standby"
            utilization     : [—]     |Q_out| / Q_max
        """
        return self._model.compute(
            Q_demand_MVAR=inputs["Q_demand_MVAR"],
            V_pu=inputs.get("V_pu", 1.0),
        )

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "STATCOM (VSC-Based Static Synchronous Compensator)",
            "ec_id": "EC186",
            "fidelity": "F1a",
            "description": "Q_out=clamp(Q_demand); P_loss=P_standby+loss_factor*|Q_out|; symmetric Q range",
            "inputs": {
                "Q_demand_MVAR": {"unit": "MVAR", "range": [-100.0, 100.0]},
                "V_pu": {"unit": "pu", "range": [0.8, 1.2], "optional": True},
            },
            "outputs": {
                "Q_out_MVAR": {"unit": "MVAR"},
                "Q_limited": {"unit": "bool"},
                "P_loss_MW": {"unit": "MW"},
                "P_standby_MW": {"unit": "MW"},
                "P_total_loss_MW": {"unit": "MW"},
                "operating_mode": {"unit": "enum"},
                "utilization": {"unit": "dimensionless"},
            },
            "params": {
                "Q_max": f"±{u['Q_max_MVAR']['value']} MVAR",
                "loss_factor": f"{u['loss_factor']['value']*100:.1f}%",
                "P_standby": f"{u['P_standby_MW']['value']} MW",
                "response_time": f"{u['response_time_ms']['value']} ms",
            },
            "source": "Hingorani & Gyugyi (2000), Understanding FACTS, IEEE Press",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for Q_req in [-120, -80, -30, 0, 50, 100, 130]:
        r = model.predict({"Q_demand_MVAR": float(Q_req)})
        print(f"Q_req={Q_req:>5} → Q_out={float(r['Q_out_MVAR']):>7.1f}  "
              f"mode={r['operating_mode']:<12}  P_total={float(r['P_total_loss_MW']):.3f} MW")
