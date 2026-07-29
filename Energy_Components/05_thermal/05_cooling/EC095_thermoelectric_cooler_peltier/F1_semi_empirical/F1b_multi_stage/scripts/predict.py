"""EC095 — Peltier TEC — F1b Multi-Stage — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import PeltierTECF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PeltierTECF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            current_stages : list or array of length n_stages — current per stage [A]
                             or scalar (same current all stages)
            T_cold : float or array [degC]
            T_hot  : float or array [degC]
        """
        I = inputs.get("current_stages", inputs.get("current", None))
        if I is None:
            raise ValueError("Provide 'current_stages' or 'current' in inputs")
        I = np.asarray(I, dtype=float)
        # If scalar, broadcast to all stages
        if I.ndim == 0 or (I.ndim == 1 and len(I) == 1):
            I = np.full(self._model.n_stages, float(I))

        T_cold = np.asarray(inputs["T_cold"], dtype=float)
        T_hot  = np.asarray(inputs["T_hot"],  dtype=float)

        return self._model.solve(I, T_cold, T_hot)

    def get_info(self) -> dict:
        return {
            "name": "Peltier TEC — Multi-Stage Cascade",
            "ec_id": "EC095",
            "fidelity": "F1b",
            "description": (
                f"Cascaded {self._model.n_stages}-stage Peltier cooler. "
                "Equal temperature partition between stages. "
                "Inter-stage T from energy balance + contact resistance. "
                "COP_cascade < COP_single at same total dT."
            ),
            "inputs": {
                "current_stages": {"unit": "A", "shape": f"n_stages={self._model.n_stages}"},
                "T_cold": {"unit": "degC", "range": [-40.0, 20.0]},
                "T_hot":  {"unit": "degC", "range": [10.0, 80.0]},
            },
            "outputs": {
                "Q_c_kw":               {"unit": "kW"},
                "W_total_kw":           {"unit": "kW"},
                "COP":                  {"unit": "-"},
                "T_inter_C":            {"unit": "degC", "note": "Inter-stage temperature"},
                "COP_single_stage_ref": {"unit": "-", "note": "Single-stage COP at same dT for comparison"},
            },
            "source": "Goldsmid (2010); Rowe (2006); Chein & Chen (2005)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    n = model._model.n_stages

    print(f"=== EC095 Peltier TEC F1b — {n}-stage cascade ===")
    print("\nCurrent sweep at T_cold=0C, T_hot=50C:")
    for I in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        I_list = [I] * n
        r = model.predict({"current_stages": I_list, "T_cold": 0.0, "T_hot": 50.0})
        print(f"  I={I:.1f}A: Q_c={float(r['Q_c_kw'])*1000:.1f}W, "
              f"COP={float(r['COP']):.3f}, "
              f"COP_single={float(r['COP_single_stage_ref']):.3f}, "
              f"T_inter={float(r['T_inter_C']):.1f}C")

    print("\ndT sweep at optimal currents:")
    for T_cold in [10.0, 0.0, -10.0, -20.0]:
        T_hot = 50.0
        I_opt = model._model.optimum_currents(T_cold, T_hot)
        r = model.predict({"current_stages": I_opt, "T_cold": T_cold, "T_hot": T_hot})
        print(f"  T_cold={T_cold:.0f}C, dT={T_hot-T_cold:.0f}K: "
              f"COP={float(r['COP']):.3f}, COP_single={float(r['COP_single_stage_ref']):.3f}")
