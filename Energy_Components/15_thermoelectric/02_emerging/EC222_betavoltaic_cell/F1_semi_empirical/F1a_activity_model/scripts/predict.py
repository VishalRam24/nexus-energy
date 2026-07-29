"""EC222 — Betavoltaic Cell — F1a — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import BetavoltaicF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BetavoltaicF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            t_years : float or array — time since deployment [years]
        returns:
            activity_Bq       : Bq
            P_beta_W          : W (total beta thermal power)
            P_out_W           : W (electrical output)
            P_out_uW          : uW
            fraction_remaining: dimensionless (A(t)/A0)
        """
        t = np.asarray(inputs["t_years"], dtype=float)
        return self._model.compute(t)

    def get_info(self) -> dict:
        return {
            "name": "Betavoltaic Cell",
            "ec_id": "EC222",
            "fidelity": "F1a",
            "description": "A(t)=A0*exp(-ln2*t/t_half); P=A*E_beta*eta_cap*eta_conv; uW scale",
            "inputs": {
                "t_years": {"unit": "years", "range": [0.0, 500.0]},
            },
            "outputs": {
                "activity_Bq":        {"unit": "Bq"},
                "P_beta_W":           {"unit": "W"},
                "P_out_W":            {"unit": "W"},
                "P_out_uW":           {"unit": "uW"},
                "fraction_remaining": {"unit": "-"},
            },
            "source": "Olsen et al. (1993) Nucl. Instrum. Methods; Sychov et al. (2008) Appl. Radiat. Isot.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    t_test = [0, 10, 25, 50, 100]
    for t in t_test:
        r = model.predict({"t_years": float(t)})
        print(f"t={t:4d}y: A={float(r['activity_Bq']):.3e} Bq, "
              f"P={float(r['P_out_uW']):.3f} uW, "
              f"remaining={float(r['fraction_remaining'])*100:.1f}%")
