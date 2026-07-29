"""EC130 — Small/Micro Hydropower — F1a Power Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import SmallMicroHydroF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SmallMicroHydroF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            flow_rate_m3s : m3/s
            net_head_m    : m (net head after losses)
            turbine_type  : 'pelton'|'francis'|'kaplan'|'auto' (default 'auto')
        """
        Q = np.asarray(inputs["flow_rate_m3s"], dtype=float)
        H = np.asarray(inputs["net_head_m"], dtype=float)
        t_type = inputs.get("turbine_type", "auto")
        return {
            "power_kw": self._model.power_kw(Q, H, t_type),
            "turbine_efficiency": self._model.turbine_efficiency(Q, t_type, H),
            "overall_efficiency": self._model.overall_efficiency(Q, t_type, H),
            "capacity_factor": self._model.capacity_factor(Q, H, t_type),
            "turbine_type": t_type if t_type != "auto" else self._model.turbine_type_for_head(float(np.mean(H))),
        }

    def get_info(self) -> dict:
        return {
            "name": "Small/Micro Hydropower",
            "ec_id": "EC130",
            "fidelity": "F1a",
            "description": "P = eta * rho * g * Q * H_net / 1000 kW; auto turbine selection (Pelton/Francis/Kaplan); 1 kW–10 MW",
            "inputs": {
                "flow_rate_m3s": {"unit": "m3/s", "range": [0.0, 1.65]},
                "net_head_m": {"unit": "m", "range": [2.0, 1800.0]},
                "turbine_type": {"unit": "str", "values": ["auto", "pelton", "francis", "kaplan"]},
            },
            "outputs": {
                "power_kw": {"unit": "kW"},
                "turbine_efficiency": {"unit": "dimensionless"},
                "overall_efficiency": {"unit": "dimensionless"},
                "capacity_factor": {"unit": "dimensionless"},
                "turbine_type": {"unit": "str"},
            },
            "source": "Penche (1998); Harvey et al. (1993)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for H, t in [(20.0, "auto"), (50.0, "francis"), (200.0, "pelton"), (5.0, "kaplan")]:
        r = model.predict({"flow_rate_m3s": 1.5, "net_head_m": H, "turbine_type": t})
        print(f"H={H:6.1f} m  type={r['turbine_type']:7s}  P={float(r['power_kw']):8.1f} kW  "
              f"eta_t={float(r['turbine_efficiency']):.3f}  CF={float(r['capacity_factor']):.3f}")
