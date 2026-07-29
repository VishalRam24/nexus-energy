"""EC183 — Circuit Breaker — F1a Loss Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import CircuitBreakerModel


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CircuitBreakerModel(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            I_A         : float or array [A]   load current (closed state)
            state       : str  "closed" or "open" (default "closed")
            I_fault_kA  : float or array [kA]  prospective fault current (default 0)
        returns:
            P_loss_W        : [W]   conduction loss
            is_overloaded   : [bool] current > I_rated
            can_interrupt   : [bool] I_fault <= interrupting rating
            E_fault_J       : [J]   I^2*R*t_clear thermal energy during fault
            thermal_rating_ok: [bool]
            state           : str
        """
        return self._model.compute(
            I_A=inputs["I_A"],
            state=inputs.get("state", "closed"),
            I_fault_kA=inputs.get("I_fault_kA", 0.0),
        )

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Circuit Breaker (MV Vacuum)",
            "ec_id": "EC183",
            "fidelity": "F1a",
            "description": "P_loss=I^2*R_cb (closed); interrupting rating: I_fault<=I_interrupt",
            "inputs": {
                "I_A": {"unit": "A", "range": [0.0, 630.0]},
                "state": {"unit": "enum", "values": ["closed", "open"]},
                "I_fault_kA": {"unit": "kA", "range": [0.0, 25.0], "optional": True},
            },
            "outputs": {
                "P_loss_W": {"unit": "W"},
                "is_overloaded": {"unit": "bool"},
                "can_interrupt": {"unit": "bool"},
                "E_fault_J": {"unit": "J"},
                "thermal_rating_ok": {"unit": "bool"},
            },
            "params": {
                "R_cb": f"{u['R_cb_ohm']['value']*1e6:.0f} uOhm",
                "I_rated": f"{u['I_rated_A']['value']} A",
                "I_interrupt": f"{u['I_interrupt_kA']['value']} kA",
                "t_clear": f"{u['t_clear_ms']['value']} ms",
                "V_rated": f"{u['V_rated_kV']['value']} kV",
            },
            "source": "ABB Circuit Breaker Application Guide (2021); IEC 62271-100",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"I_A": 400.0, "state": "closed", "I_fault_kA": 15.0})
    print(f"P_loss={float(r['P_loss_W'])*1000:.2f} mW  "
          f"overload={r['is_overloaded']}  "
          f"can_interrupt={r['can_interrupt']}")
