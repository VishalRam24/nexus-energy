"""EC185 -- SVC -- F1b Conduction+Switching -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import SVCF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SVCF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            Q_demand_MVAR : float or array [-50, 100]
            V_pu          : float or array [0.85, 1.15] (default 1.0)
        returns:
            Q_out_MVAR, Q_effective_MVAR, Q_limited,
            P_thyristor_MW, P_reactor_MW, P_ESR_TSC_MW, P_cooling_MW, P_loss_MW,
            I_rms_A, operating_mode, utilization
        """
        Q = np.asarray(inputs.get("Q_demand_MVAR", 0.0), dtype=float)
        V = inputs.get("V_pu", 1.0)
        return self._model.compute(Q, V)

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Static VAR Compensator (SVC) — Conduction + Switching Losses",
            "ec_id": "EC185",
            "fidelity": "F1b",
            "description": (
                "TCR mode: thyristor + reactor copper losses; "
                "TSC mode: capacitor ESR loss; "
                "cooling always-on; V^2 output scaling."
            ),
            "inputs": {
                "Q_demand_MVAR": {"unit": "MVAR", "range": [-50, 100]},
                "V_pu": {"unit": "pu", "range": [0.85, 1.15], "default": 1.0},
            },
            "outputs": {
                "Q_out_MVAR": {"unit": "MVAR"},
                "Q_effective_MVAR": {"unit": "MVAR"},
                "P_thyristor_MW": {"unit": "MW"},
                "P_reactor_MW": {"unit": "MW"},
                "P_ESR_TSC_MW": {"unit": "MW"},
                "P_cooling_MW": {"unit": "MW"},
                "P_loss_MW": {"unit": "MW"},
                "I_rms_A": {"unit": "A"},
            },
            "params": {
                "Q_max": f"{u['Q_max_MVAR']['value']} MVAR",
                "Q_min": f"{u['Q_min_MVAR']['value']} MVAR",
                "n_thyristors_series": u["n_thyristors_series"]["value"],
                "V_T0": f"{u['thyristor_V_T0']['value']} V",
                "r_T": f"{u['thyristor_r_T']['value']} Ohm",
            },
            "source": "Hingorani & Gyugyi (2000); Cigre TB 25; IEEE Std 1031-2011",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("=== EC185 F1b SVC Conduction+Switching ===")
    for Q in [-50, -25, 0, 25, 50, 100]:
        r = model.predict({"Q_demand_MVAR": float(Q), "V_pu": 1.0})
        print(f"  Q={Q:+4d} MVAR: P_loss={float(r['P_loss_MW']):.3f} MW  "
              f"P_thy={float(r['P_thyristor_MW']):.3f} MW  "
              f"P_rctr={float(r['P_reactor_MW']):.3f} MW  "
              f"mode={r['operating_mode']}")
