"""EC186 -- STATCOM -- F1b Conduction+Switching -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import STATCOMF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = STATCOMF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            Q_demand_MVAR : float or array [-100, 100]
            V_pu          : float or array [0.8, 1.2] (default 1.0)
            T_j           : float [degC] junction temperature (default from params)
        returns:
            Q_out_MVAR, Q_limited, P_cond_MW, P_sw_MW, P_diode_MW,
            P_transformer_MW, P_standby_MW, P_total_loss_MW,
            I_rms_A, r_ce_Ohm, operating_mode, utilization
        """
        Q = np.asarray(inputs.get("Q_demand_MVAR", 0.0), dtype=float)
        V = inputs.get("V_pu", 1.0)
        T = inputs.get("T_j", None)
        return self._model.compute(Q, V, T)

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "STATCOM (VSC) — IGBT Conduction + Switching Losses",
            "ec_id": "EC186",
            "fidelity": "F1b",
            "description": (
                "IGBT conduction: P=V_ce0*I_avg+r_ce(T)*I_rms^2; "
                "switching: P=E_sw*f_sw*(V/V_ref)*(I/I_ref); "
                "diode ~30% of IGBT; transformer copper losses."
            ),
            "inputs": {
                "Q_demand_MVAR": {"unit": "MVAR", "range": [-100, 100]},
                "V_pu": {"unit": "pu", "range": [0.8, 1.2], "default": 1.0},
                "T_j": {"unit": "degC", "range": [25, 150], "default": u["T_j"]["value"]},
            },
            "outputs": {
                "Q_out_MVAR": {"unit": "MVAR"},
                "P_cond_MW": {"unit": "MW"},
                "P_sw_MW": {"unit": "MW"},
                "P_diode_MW": {"unit": "MW"},
                "P_transformer_MW": {"unit": "MW"},
                "P_standby_MW": {"unit": "MW"},
                "P_total_loss_MW": {"unit": "MW"},
                "I_rms_A": {"unit": "A"},
            },
            "params": {
                "Q_max": f"±{u['Q_max_MVAR']['value']} MVAR",
                "V_ce0": f"{u['IGBT_V_ce0']['value']} V",
                "r_ce_ref": f"{u['IGBT_r_ce']['value']*1000:.1f} mOhm at {u['IGBT_T_j_ref']['value']} C",
                "f_sw": f"{u['f_sw_Hz']['value']} Hz",
                "n_devices": self._model.n_devices,
            },
            "source": "Hingorani & Gyugyi (2000); IEC 62927:2018; Semikron IGBT Manual (2021)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("=== EC186 F1b STATCOM Conduction+Switching ===")
    for Q in [-100, -50, 0, 50, 100]:
        for T_j in [25, 125]:
            r = model.predict({"Q_demand_MVAR": float(Q), "T_j": float(T_j)})
            print(f"  Q={Q:+4d} MVAR T_j={T_j}C: P_total={float(r['P_total_loss_MW']):.3f} MW  "
                  f"P_cond={float(r['P_cond_MW']):.3f}  "
                  f"P_sw={float(r['P_sw_MW']):.3f}  "
                  f"mode={r['operating_mode']}")
