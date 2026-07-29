"""EC091 — Vapor Compression Chiller — F1a COP Map — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import ChillerF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ChillerF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            T_chw_supply (degC): Chilled water supply / evaporator temperature [4–12]
            T_cond (degC):       Condenser temperature [25–45]
            part_load_ratio (-): Part-load ratio [0.1–1.0], default 1.0

        returns:
            cop (-)
            cooling_kw (kW)
            electrical_kw (kW)
            heat_rejection_kw (kW)
        """
        T_evap = np.asarray(inputs["T_chw_supply"], dtype=float)
        T_cond = np.asarray(inputs["T_cond"], dtype=float)
        plr = np.asarray(inputs.get("part_load_ratio", 1.0), dtype=float)
        return {
            "cop": self._model.cop(T_evap, T_cond, plr),
            "cooling_kw": self._model.cooling_power(plr),
            "electrical_kw": self._model.compressor_power(T_evap, T_cond, plr),
            "heat_rejection_kw": self._model.heat_rejection(T_evap, T_cond, plr),
        }

    def get_info(self) -> dict:
        return {
            "name": "Vapor Compression Chiller",
            "ec_id": "EC091",
            "fidelity": "F1a",
            "description": (
                "COP = eta_Carnot * T_evap/(T_cond - T_evap), "
                "part-load: COP(PLR) = COP_full * (c1 + c2*PLR + c3*PLR^2)"
            ),
            "inputs": {
                "T_chw_supply": {"unit": "degC", "range": [4.0, 12.0]},
                "T_cond": {"unit": "degC", "range": [25.0, 45.0]},
                "part_load_ratio": {"unit": "-", "range": [0.1, 1.0], "default": 1.0},
            },
            "outputs": {
                "cop": {"unit": "dimensionless"},
                "cooling_kw": {"unit": "kW"},
                "electrical_kw": {"unit": "kW"},
                "heat_rejection_kw": {"unit": "kW"},
            },
            "source": "Gordon & Ng (2000), Cool Thermodynamics",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_chw_supply": 5.0, "T_cond": 35.0, "part_load_ratio": 1.0})
    print(f"At nominal: COP={float(r['cop']):.2f}, Q={float(r['cooling_kw']):.1f}kW, "
          f"W={float(r['electrical_kw']):.1f}kW, Q_rej={float(r['heat_rejection_kw']):.1f}kW")
