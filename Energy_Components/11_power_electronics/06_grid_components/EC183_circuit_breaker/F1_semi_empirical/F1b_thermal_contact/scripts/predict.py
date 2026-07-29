"""EC183 -- Circuit Breaker -- F1b Thermal Contact -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import CircuitBreakerF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CircuitBreakerF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            I_A           : float or array [0, 800]  Load current [A]
            state         : str "closed" or "open"   (default "closed")
            T_contact     : float or array [degC]    (default 50)
            T_ambient     : float or array [degC]    (default 20)
            I_fault_kA    : float or array [0, 25]   (default 0)
        returns:
            P_cond_W, P_aux_W, P_total_W, R_contact_Ohm, F_skin,
            I_max_thermal_A, thermal_margin, is_overloaded,
            can_interrupt, E_fault_J
        """
        I = np.asarray(inputs.get("I_A", 0.0), dtype=float)
        state = inputs.get("state", "closed")
        T_c = inputs.get("T_contact", 50.0)
        T_a = inputs.get("T_ambient", 20.0)
        I_f = inputs.get("I_fault_kA", 0.0)
        return self._model.compute(I, state, T_c, T_a, I_f)

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Circuit Breaker (Thermal Contact + Ampacity + Skin Effect)",
            "ec_id": "EC183",
            "fidelity": "F1b",
            "description": (
                "R(T)=R_ref*(1+alpha*(T-T_ref))*F_skin; "
                "IEC 62271-1 ampacity derating; auxiliary standby power."
            ),
            "inputs": {
                "I_A": {"unit": "A", "range": [0, 800]},
                "state": {"unit": "str", "values": ["closed", "open"]},
                "T_contact": {"unit": "degC", "range": [-10, 130], "default": 50},
                "T_ambient": {"unit": "degC", "range": [-10, 50], "default": 20},
                "I_fault_kA": {"unit": "kA", "range": [0, 25], "default": 0},
            },
            "outputs": {
                "P_cond_W": {"unit": "W"},
                "P_aux_W": {"unit": "W"},
                "P_total_W": {"unit": "W"},
                "R_contact_Ohm": {"unit": "Ohm"},
                "F_skin": {"unit": "dimensionless"},
                "I_max_thermal_A": {"unit": "A"},
                "thermal_margin": {"unit": "dimensionless"},
                "is_overloaded": {"unit": "bool"},
                "can_interrupt": {"unit": "bool"},
                "E_fault_J": {"unit": "J"},
            },
            "params": {
                "R_ref": f"{u['R_cb_ohm']['value']*1e6:.0f} uOhm at {u['T_ref']['value']} C",
                "alpha_contact": f"{u['alpha_contact']['value']} /K",
                "T_max_contact": f"{u['T_max_contact']['value']} degC",
                "P_aux": f"{u['P_aux_W']['value']} W",
                "F_skin_precomputed": f"{self._model.F_skin:.4f}",
            },
            "source": "IEC 62271-100:2021; IEC 62271-1; Greenwood (1991)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("=== EC183 F1b Circuit Breaker Thermal Contact ===")
    print(f"  F_skin = {model._model.F_skin:.4f}")
    for T_c in [20, 50, 80, 105]:
        r = model.predict({"I_A": 630.0, "T_contact": T_c, "T_ambient": 20.0})
        print(f"  T_contact={T_c}C: R={float(r['R_contact_Ohm'])*1e6:.1f} uOhm  "
              f"P_cond={float(r['P_cond_W']):.2f} W  "
              f"P_total={float(r['P_total_W']):.2f} W  "
              f"I_max={float(r['I_max_thermal_A']):.1f} A")
