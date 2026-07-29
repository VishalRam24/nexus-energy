"""
EC216 -- Thermoelectric Generator (TEG) -- F2a Coupled Thermal-Electrical -- predict interface.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import TEG_CoupledF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)


class ComponentModel:
    component_id = "EC216"
    component_name = "Thermoelectric Generator (TEG)"
    fidelity = "F2a -- Coupled Thermal-Electrical Model"
    version = "1.0.0"

    def __init__(self, params=None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = TEG_CoupledF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            T_hot_K   : float  -- heat source temperature [K]
            T_cold_K  : float  -- heat sink temperature [K]
            R_load    : float  -- external load resistance [ohm] (optional, default=matched)
        """
        T_hot = inputs.get("T_hot_K", 473.15)
        T_cold = inputs.get("T_cold_K", 300.0)
        R_load = inputs.get("R_load", None)

        if R_load is None:
            R_load = self._model.matched_load_resistance(T_hot, T_cold)

        return self._model.solve_steady_state(T_hot, T_cold, R_load)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_hot_K": {"unit": "K", "range": [300, 600], "note": "Heat source temperature"},
                "T_cold_K": {"unit": "K", "range": [250, 350], "note": "Heat sink temperature"},
                "R_load": {"unit": "ohm", "range": [0.01, 100], "note": "External load resistance (optional, default=matched)"},
            },
            "outputs": {
                "P": "W (electrical power output)",
                "V": "V (terminal voltage)",
                "I": "A (current)",
                "efficiency": "- (P/Q_h)",
                "Q_h": "W (hot-side heat absorption)",
                "Q_c": "W (cold-side heat rejection)",
                "V_oc": "V (open-circuit voltage)",
                "R_int": "ohm (internal resistance)",
                "T_h_junction": "K (hot junction temperature)",
                "T_c_junction": "K (cold junction temperature)",
                "eta_carnot": "- (Carnot efficiency limit)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    r = m.predict({"T_hot_K": 473.15, "T_cold_K": 300.0})
    print(f"P={r['P']:.3f} W, V={r['V']:.3f} V, I={r['I']:.3f} A, "
          f"eta={r['efficiency']:.4f}, Q_h={r['Q_h']:.2f} W")
