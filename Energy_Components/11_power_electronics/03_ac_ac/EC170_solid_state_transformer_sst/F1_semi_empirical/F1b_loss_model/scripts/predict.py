"""EC170 -- Solid State Transformer -- F1b Three-Stage Loss Model -- Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import SolidStateTransformerF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SolidStateTransformerF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            p_out        : float or array [W]  output active power
            power_factor : float or array      (default 1.0)
        returns:
            efficiency, p_loss_w, p_stage1_w, p_stage2_w, p_stage3_w, t_j_degc
        """
        p_out = np.asarray(inputs["p_out"], dtype=float)
        pf = np.asarray(inputs.get("power_factor", 1.0), dtype=float)

        breakdown = self._model.loss_breakdown(p_out, pf)
        eta = self._model.efficiency(p_out, pf)
        p_loss = self._model.total_losses(p_out, pf)
        t_j = self._model.junction_temperature(p_out, pf)

        return {
            "efficiency": eta,
            "p_loss_w": p_loss,
            "t_j_degc": t_j,
            **breakdown,
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Solid State Transformer (SST)",
            "ec_id": "EC170",
            "fidelity": "F1b",
            "description": (
                "Three-stage SST loss model: Stage 1 (front-end AC-DC H-bridge), "
                "Stage 2 (isolated DAB DC-DC with transformer copper+core+switching losses), "
                "Stage 3 (output DC-AC H-bridge)."
            ),
            "inputs": {
                "p_out": {"unit": "W", "range": [0.0, 12000.0]},
                "power_factor": {"unit": "dimensionless", "range": [0.6, 1.0]},
            },
            "outputs": {
                "efficiency": {"unit": "dimensionless"},
                "p_loss_w": {"unit": "W"},
                "t_j_degc": {"unit": "degC"},
                "p_stage1_w": {"unit": "W"},
                "p_stage2_w": {"unit": "W"},
                "p_stage3_w": {"unit": "W"},
            },
            "params": {
                "P_rated": f"{u['P_rated']['value']/1000:.0f} kW",
                "V_hv": f"{u['V_hv']['value']} V",
                "V_lv": f"{u['V_lv']['value']} V",
                "turns_ratio": str(u["turns_ratio"]["value"]),
            },
            "source": "Krismer & Kolar (2012); She et al. (2013)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"p_out": 8000.0})
    print(f"eta={float(r['efficiency'])*100:.2f}%  P_loss={float(r['p_loss_w']):.1f}W  "
          f"T_j={float(r['t_j_degc']):.1f}°C")
    print(f"  Stage1={float(r['p_stage1_w']):.1f}W  "
          f"Stage2={float(r['p_stage2_w']):.1f}W  "
          f"Stage3={float(r['p_stage3_w']):.1f}W")
