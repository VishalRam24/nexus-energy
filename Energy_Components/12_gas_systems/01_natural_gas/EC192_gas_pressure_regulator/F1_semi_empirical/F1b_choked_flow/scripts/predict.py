"""EC192 — Gas Pressure Regulator — F1b Choked Flow — Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import GasPressureRegulatorF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = GasPressureRegulatorF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict regulator flow with correct choked-flow treatment.

        Parameters
        ----------
        inputs : dict
            P_up_bar     : float (bar)
            P_down_bar   : float (bar)
            T_up_K       : float (K, default 288.15)
            Z            : float (default 0.9)
            Cv           : float (optional, default from params)
            valve_travel : float (0.1-1.0, default 1.0)
        """
        P_up = inputs.get("P_up_bar", 70.0)
        P_down = inputs.get("P_down_bar", 10.0)
        T_up = inputs.get("T_up_K", 288.15)
        Z = inputs.get("Z", 0.9)
        Cv = inputs.get("Cv", None)
        travel = inputs.get("valve_travel", 1.0)
        return self._model.compute(P_up, P_down, T_up, Z, Cv, travel)

    def get_info(self) -> dict:
        return {
            "name": "Gas Pressure Regulator",
            "ec_id": "EC192",
            "fidelity": "F1b",
            "description": (
                "ISA Cv gas flow with correct choked-flow: at choke, uses ΔP_choke "
                "(= Fk*Xt*P_up) in sqrt, not actual ΔP. Valve travel → effective Cv. "
                "JT temperature correction included."
            ),
            "inputs": {
                "P_up_bar": {"unit": "bar", "range": [1, 200]},
                "P_down_bar": {"unit": "bar", "range": [0.5, 100]},
                "T_up_K": {"unit": "K", "range": [240, 330], "default": 288.15},
                "Z": {"unit": "dimensionless", "range": [0.7, 1.0], "default": 0.9},
                "valve_travel": {"unit": "dimensionless", "range": [0.1, 1.0], "default": 1.0},
            },
            "outputs": {
                "flow_std_m3_per_h": {"unit": "m3/h"},
                "flow_kg_per_s": {"unit": "kg/s"},
                "T_downstream_K": {"unit": "K"},
                "is_choked": {"unit": "bool"},
                "expansion_factor_Y": {"unit": "dimensionless"},
            },
            "source": "ANSI/ISA-75.01.01-2012; Driskell (1983)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    # Test subcritical
    r1 = model.predict({"P_up_bar": 70.0, "P_down_bar": 60.0, "T_up_K": 288.15})
    print("Subcritical (70 → 60 bar):")
    for k, v in r1.items():
        vv = np.atleast_1d(v)[0]
        print(f"  {k} = {vv}")
    # Test choked
    r2 = model.predict({"P_up_bar": 70.0, "P_down_bar": 10.0, "T_up_K": 288.15})
    print("\nChoked (70 → 10 bar):")
    for k, v in r2.items():
        vv = np.atleast_1d(v)[0]
        print(f"  {k} = {vv}")
