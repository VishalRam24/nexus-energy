"""EC130 — Small/Micro Hydropower — F1b Head-Flow — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import SmallMicroHydroF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SmallMicroHydroF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            flow_rate_m3s : float or array [m3/s]
            gross_head_m  : float or array [m]
            turbine_type  : 'pelton'|'francis'|'kaplan'|'auto' (default 'auto')
            T_water_C     : water temperature [°C] (optional)
        returns:
            power_kw, efficiency, capacity_factor, net_head_m,
            head_loss_fraction, cavitation_derate, turbine_type_used
        """
        Q = np.asarray(inputs["flow_rate_m3s"], dtype=float)
        H_gross = np.asarray(inputs["gross_head_m"], dtype=float)
        ttype = inputs.get("turbine_type", "auto")
        T_water = inputs.get("T_water_C", None)

        m = self._model
        Q_avail = np.maximum(Q - m.Q_eco, 0.0)
        H_net = m.net_head(H_gross, Q_avail)

        if ttype == "auto":
            ttype_used = m.turbine_type_for_head(float(np.mean(H_net)))
        else:
            ttype_used = ttype

        return {
            "power_kw": m.power_kw(Q, H_gross, ttype, T_water),
            "efficiency": m.overall_efficiency(Q_avail, H_net, ttype),
            "capacity_factor": m.capacity_factor(Q, H_gross, ttype, T_water),
            "net_head_m": H_net,
            "head_loss_fraction": m.penstock_head_loss_fraction(Q_avail),
            "cavitation_derate": m.cavitation_derate(H_net, ttype),
            "turbine_type_used": ttype_used,
        }

    def get_info(self) -> dict:
        return {
            "name": "Small/Micro Hydropower (Head-Flow)",
            "ec_id": "EC130",
            "fidelity": "F1b",
            "description": (
                "2D hill chart per turbine type; Q^2 variable head losses; "
                "cavitation derate; water temperature density correction; "
                "ecological flow constraint."
            ),
            "inputs": {
                "flow_rate_m3s": {"unit": "m3/s", "range": [0, 10]},
                "gross_head_m": {"unit": "m", "range": [2, 1800]},
                "turbine_type": {"values": ["pelton", "francis", "kaplan", "auto"]},
                "T_water_C": {"unit": "degC", "range": [1, 25], "optional": True},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "efficiency": {"unit": "dimensionless"},
                "capacity_factor": {"unit": "dimensionless"},
                "cavitation_derate": {"unit": "dimensionless"},
            },
            "source": "Penche (1998); Harvey et al. (1993); Fraenkel et al. (1991)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("=== EC130 Small/Micro Hydro F1b ===\n")
    # Phase 7: use 30% flow for part-load (not 70% which hits P_rated cap)
    Q_30 = 0.30 * model._model.Q_design
    Q_full = model._model.Q_design
    print(f"Part-load (30% flow = {Q_30:.2f} m3/s, H=45m):")
    r = model.predict({"flow_rate_m3s": Q_30, "gross_head_m": 45.0})
    print(f"  P={float(r['power_kw']):.1f} kW, eta={float(np.mean(r['efficiency'])):.3f}")
    print(f"\nDesign point ({Q_full:.2f} m3/s, H=45m):")
    r = model.predict({"flow_rate_m3s": Q_full, "gross_head_m": 45.0})
    print(f"  P={float(r['power_kw']):.1f} kW, eta={float(np.mean(r['efficiency'])):.3f}")
