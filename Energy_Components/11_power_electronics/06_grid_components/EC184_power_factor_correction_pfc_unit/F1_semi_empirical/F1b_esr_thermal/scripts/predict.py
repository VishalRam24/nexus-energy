"""EC184 -- PFC Unit -- F1b ESR Thermal -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import PFCUnitF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PFCUnitF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            P_kW               : float or array [kW]
            pf_initial         : float or array [0.5, 0.99]
            T_cap              : float or array [degC] (default 25)
            pf_target          : float (default from params)
            Q_comp_override_kVAR : float or array (optional)
        returns:
            Q_load_kVAR, Q_required_kVAR, Q_compensated_kVAR, Q_residual_kVAR,
            pf_achieved, P_ESR_kW, P_dielectric_kW, P_loss_kW, I_cap_A,
            ESR_Ohm, tan_delta, bank_utilization, Q_rated_available_kVAR
        """
        P = np.asarray(inputs.get("P_kW", 1000.0), dtype=float)
        pf1 = np.asarray(inputs.get("pf_initial", 0.8), dtype=float)
        T = inputs.get("T_cap", 25.0)
        pft = inputs.get("pf_target", None)
        Qov = inputs.get("Q_comp_override_kVAR", None)
        return self._model.compute(P, pf1, T, pft, Qov)

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Power Factor Correction Unit (ESR + Thermal)",
            "ec_id": "EC184",
            "fidelity": "F1b",
            "description": (
                "ESR-based conduction loss: P_ESR=ESR(T)*I^2; "
                "dielectric loss: P_diel=tan_delta(T)*Q_comp; "
                "IEC 60831-1 thermal derating above T_max."
            ),
            "inputs": {
                "P_kW": {"unit": "kW", "range": [0, 10000]},
                "pf_initial": {"unit": "dimensionless", "range": [0.5, 0.99]},
                "T_cap": {"unit": "degC", "range": [-10, 80], "default": 25},
                "pf_target": {"unit": "dimensionless", "default": u["pf_target"]["value"]},
            },
            "outputs": {
                "Q_compensated_kVAR": {"unit": "kVAR"},
                "pf_achieved": {"unit": "dimensionless"},
                "P_ESR_kW": {"unit": "kW"},
                "P_dielectric_kW": {"unit": "kW"},
                "P_loss_kW": {"unit": "kW"},
                "I_cap_A": {"unit": "A"},
                "ESR_Ohm": {"unit": "Ohm"},
                "tan_delta": {"unit": "dimensionless"},
                "Q_rated_available_kVAR": {"unit": "kVAR"},
            },
            "params": {
                "Q_rated": f"{u['Q_rated_kVAR']['value']} kVAR",
                "ESR_ref": f"{u['ESR_ref_ohm']['value']*1000:.1f} mOhm at {u['T_ref']['value']} C",
                "ESR_alpha": f"{u['ESR_alpha']['value']} /K",
                "tan_delta_ref": f"{u['tan_delta_ref']['value']}",
                "T_max": f"{u['T_max_C']['value']} degC",
            },
            "source": "IEEE Std 1036-2010; IEC 60831-1:2014",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("=== EC184 F1b PFC ESR Thermal ===")
    for T in [25, 45, 70, 75]:
        r = model.predict({"P_kW": 5000.0, "pf_initial": 0.75, "T_cap": T})
        print(f"  T={T}C: Q_comp={float(r['Q_compensated_kVAR']):.0f} kVAR  "
              f"pf={float(r['pf_achieved']):.4f}  "
              f"P_ESR={float(r['P_ESR_kW'])*1000:.2f} W  "
              f"P_diel={float(r['P_dielectric_kW'])*1000:.2f} W  "
              f"ESR={float(r['ESR_Ohm'])*1000:.2f} mOhm")
