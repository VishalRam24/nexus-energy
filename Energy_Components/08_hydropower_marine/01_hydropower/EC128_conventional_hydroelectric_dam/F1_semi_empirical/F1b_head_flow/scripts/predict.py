"""EC128 — Conventional Hydro Dam — F1b Head-Flow — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import HydroelectricDamF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = HydroelectricDamF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            flow_rate_m3s : float or array [m3/s]
            head_m        : float or array [m]
            turbine_type  : "francis" / "kaplan" / "pelton" (default "francis")
        returns:
            power_kw, efficiency, specific_speed, flow_ratio
        """
        Q = np.asarray(inputs["flow_rate_m3s"], dtype=float)
        H = np.asarray(inputs["head_m"], dtype=float)
        ttype = inputs.get("turbine_type", "francis")

        return {
            "power_kw": self._model.power_kw(Q, H, ttype),
            "efficiency": self._model.overall_efficiency(Q, H, ttype),
            "specific_speed": self._model.specific_speed(Q, H, ttype),
            "flow_ratio": self._model.flow_ratio(Q, ttype),
        }

    def get_info(self) -> dict:
        return {
            "name": "Conventional Hydroelectric Dam (Head-Flow)",
            "ec_id": "EC128",
            "fidelity": "F1b",
            "description": (
                "2D hill chart: eta(q,h) = eta_peak*(1-k_q*(q-1)^2)*(1-k_h*(h-1)^2); "
                "Francis/Kaplan/Pelton turbine types; environmental flow constraint"
            ),
            "inputs": {
                "flow_rate_m3s": {"unit": "m3/s", "range": [0, 200]},
                "head_m": {"unit": "m", "range": [5, 1800]},
                "turbine_type": {"values": ["francis", "kaplan", "pelton"], "default": "francis"},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "efficiency": {"unit": "dimensionless"},
                "specific_speed": {"unit": "dimensionless"},
                "flow_ratio": {"unit": "dimensionless"},
            },
            "source": "Dixon & Hall (2014); IEC 60041",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for ttype in ["francis", "kaplan", "pelton"]:
        # Use design-point values
        defaults = {"francis": (50, 100), "kaplan": (100, 20), "pelton": (10, 600)}
        Q, H = defaults[ttype]
        r = model.predict({"flow_rate_m3s": Q, "head_m": H, "turbine_type": ttype})
        print(
            f"{ttype:>7s}: Q={Q}m3/s H={H}m -> P={float(r['power_kw']):.0f}kW  "
            f"eta={float(r['efficiency']):.4f}  "
            f"Ns={float(r['specific_speed']):.4f}  "
            f"q={float(r['flow_ratio']):.2f}"
        )
