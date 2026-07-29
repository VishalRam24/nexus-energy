"""EC129 — Run-of-River Hydropower — F1b Head-Flow — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import RunOfRiverF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = RunOfRiverF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            flow_rate_m3s : float or array [m3/s]
            gross_head_m  : float or array [m]
            T_water_C     : water temperature [°C] (optional, affects density)
        returns:
            power_kw, efficiency, capacity_factor, net_head_m,
            head_loss_fraction, flow_ratio, cavitation_derate
        """
        Q = np.asarray(inputs["flow_rate_m3s"], dtype=float)
        H_gross = np.asarray(inputs["gross_head_m"], dtype=float)
        T_water = inputs.get("T_water_C", None)

        m = self._model
        Q_avail = np.maximum(Q - m.Q_eco, 0.0)
        H_net = m.net_head(H_gross, Q_avail)

        return {
            "power_kw": m.power_kw(Q, H_gross, T_water),
            "efficiency": m.overall_efficiency(Q_avail, H_net),
            "capacity_factor": m.capacity_factor(Q, H_gross, T_water),
            "net_head_m": H_net,
            "head_loss_fraction": m.penstock_head_loss_fraction(Q_avail),
            "flow_ratio": m.flow_ratio(Q_avail),
            "cavitation_derate": m.cavitation_derate(H_net),
        }

    def get_info(self) -> dict:
        return {
            "name": "Run-of-River Hydropower (Head-Flow)",
            "ec_id": "EC129",
            "fidelity": "F1b",
            "description": (
                "2D hill chart eta(q,h); variable Q^2 penstock losses; "
                "cavitation derate; ecological flow; water temperature density."
            ),
            "inputs": {
                "flow_rate_m3s": {"unit": "m3/s", "range": [0, 200]},
                "gross_head_m": {"unit": "m", "range": [1, 25]},
                "T_water_C": {"unit": "degC", "range": [1, 25], "optional": True},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "efficiency": {"unit": "dimensionless"},
                "capacity_factor": {"unit": "dimensionless"},
                "net_head_m": {"unit": "m"},
                "cavitation_derate": {"unit": "dimensionless"},
            },
            "source": "Penche (1998); Gordon (2001); IEC 60041",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("=== EC129 Run-of-River F1b Head-Flow ===\n")
    print("Part-load power curve (H_gross=8.5m):")
    for Q in [20, 40, 60, 75, 90, 100]:
        r = model.predict({"flow_rate_m3s": float(Q), "gross_head_m": 8.5})
        print(f"  Q={Q:>4d} m3/s  P={float(r['power_kw']):>6.0f} kW  "
              f"eta={float(np.mean(r['efficiency'])):.3f}  "
              f"CF={float(np.mean(r['capacity_factor'])):.3f}")
