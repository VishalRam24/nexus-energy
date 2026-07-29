"""EC122 — Pumped Hydro Storage — F1b Head Variation — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import PHSF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PHSF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            SOC             : float or array [0-1]
            flow_rate_m3s   : float or array [m3/s]
            mode            : "charge" or "discharge" (default "discharge")
        returns:
            power_kw, effective_head_m, friction_loss_m, efficiency, round_trip_efficiency
        """
        soc = np.asarray(inputs["SOC"], dtype=float)
        Q = np.asarray(inputs["flow_rate_m3s"], dtype=float)
        mode = inputs.get("mode", "discharge")

        return {
            "power_kw": self._model.power(soc, Q, mode),
            "effective_head_m": self._model.effective_head(soc),
            "friction_loss_m": self._model.friction_loss(Q),
            "efficiency": self._model.efficiency(soc, Q, mode),
            "round_trip_efficiency": self._model.round_trip_efficiency(soc, Q),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Pumped Hydro Storage (Head Variation)",
            "ec_id": "EC122",
            "fidelity": "F1b",
            "description": (
                "Variable head: H(SOC) = H_min + SOC*(H_max-H_min); "
                "Darcy-Weisbach penstock friction: h_f = f*L*v^2/(2*D*g)"
            ),
            "inputs": {
                "SOC": {"unit": "dimensionless", "range": [0.0, 1.0]},
                "flow_rate_m3s": {"unit": "m3/s", "range": [0.0, 100.0]},
                "mode": {"values": ["charge", "discharge"]},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "effective_head_m": {"unit": "m"},
                "friction_loss_m": {"unit": "m"},
                "efficiency": {"unit": "dimensionless"},
                "round_trip_efficiency": {"unit": "dimensionless"},
            },
            "params": {
                "H_max": f"{u['H_max']['value']} m",
                "H_min": f"{u['H_min']['value']} m",
                "penstock": f"L={u['penstock_length']['value']}m, D={u['penstock_diameter']['value']}m",
                "eta_turbine": u["eta_turbine"]["value"],
                "eta_pump": u["eta_pump"]["value"],
            },
            "source": "Rehman et al. (2015); Mosonyi (1991); Munson et al. (2013)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for soc in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for mode in ["discharge", "charge"]:
            r = model.predict({"SOC": soc, "flow_rate_m3s": 50.0, "mode": mode})
            print(
                f"SOC={soc:.2f} {mode:>9s}: P={float(r['power_kw']):>10.1f} kW  "
                f"H_eff={float(r['effective_head_m']):.1f} m  "
                f"h_f={float(r['friction_loss_m']):.2f} m  "
                f"eta={float(r['efficiency']):.4f}  "
                f"RTE={float(r['round_trip_efficiency']):.4f}"
            )
