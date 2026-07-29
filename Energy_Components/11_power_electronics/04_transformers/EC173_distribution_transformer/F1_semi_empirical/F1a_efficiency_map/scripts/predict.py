"""EC173 -- Distribution Transformer -- F1a Efficiency Map -- Predict Interface"""
import json, sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import DistributionTransformerF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = DistributionTransformerF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            load_fraction : float or array [-]   S_actual / S_rated (0 to 1.5)
            power_factor  : float or array [-]   load power factor (default 1.0)
            v_in          : float or array [V]   primary voltage (optional)
        returns:
            efficiency       : float or array
            p_out_w          : float or array [W]
            p_in_w           : float or array [W]
            p_losses_w       : float or array [W]
            p_core_w         : float            constant core losses
            p_copper_w       : float or array [W]
            v_out            : float or array [V]  (only if v_in provided)
        """
        plr = np.asarray(inputs["load_fraction"], dtype=float)
        pf = np.asarray(inputs.get("power_factor", 1.0), dtype=float)
        v_in = inputs.get("v_in", None)

        eta = self._model.efficiency(plr, pf)
        p_out = self._model.output_power(plr, pf)
        p_in = self._model.input_power(plr, pf)
        p_loss = self._model.losses(plr)
        p_core = self._model.core_losses()
        p_cu = self._model.copper_losses(plr)

        result = {
            "efficiency": eta,
            "p_out_w": p_out,
            "p_in_w": p_in,
            "p_losses_w": p_loss,
            "p_core_w": np.full_like(plr, p_core),
            "p_copper_w": p_cu,
        }
        if v_in is not None:
            result["v_out"] = self._model.output_voltage(np.asarray(v_in, dtype=float))
        return result

    def get_info(self) -> dict:
        u = self.params["unit"]
        m = self._model
        return {
            "name": "Distribution Transformer",
            "ec_id": "EC173",
            "fidelity": "F1a",
            "description": "eta = P_out/(P_out+P_core+P_cu*PLR^2); P_core=0.2%, P_cu=1.0% S_rated",
            "inputs": {
                "load_fraction": {"unit": "dimensionless", "range": [0.0, 1.5]},
                "power_factor": {"unit": "dimensionless", "range": [0.7, 1.0], "optional": True},
                "v_in": {"unit": "V", "optional": True},
            },
            "outputs": {
                "efficiency": {"unit": "dimensionless"},
                "p_out_w": {"unit": "W"},
                "p_in_w": {"unit": "W"},
                "p_losses_w": {"unit": "W"},
                "p_core_w": {"unit": "W"},
                "p_copper_w": {"unit": "W"},
            },
            "params": {
                "S_rated_VA": u["s_rated"]["value"],
                "P_core_W": m.P_core,
                "P_cu_W": m.P_cu,
                "peak_eta_load": round(float(m.peak_efficiency_load()), 4),
                "turns_ratio": u["n_turns"]["value"],
            },
            "source": "IEC 60076-1:2011; IEEE C57.12.00-2015",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(f"Peak eta at PLR = {model._model.peak_efficiency_load():.3f}")
    for plr in [0.1, 0.25, 0.447, 0.5, 0.75, 1.0, 1.2]:
        r = model.predict({"load_fraction": plr, "power_factor": 0.9})
        print(f"PLR={plr:.3f}: eta={float(r['efficiency'])*100:.4f}%  "
              f"P_out={float(r['p_out_w'])/1000:.1f}kW  "
              f"P_core={float(r['p_core_w'])/1000:.2f}kW  "
              f"P_cu={float(r['p_copper_w'])/1000:.3f}kW")
