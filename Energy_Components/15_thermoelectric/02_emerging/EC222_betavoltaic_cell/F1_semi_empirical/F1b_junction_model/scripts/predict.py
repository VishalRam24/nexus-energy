"""EC222 — Betavoltaic Cell — F1b Junction Model — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import BetavoltaicF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BetavoltaicF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict betavoltaic cell performance with junction electrical model.

        Parameters
        ----------
        inputs : dict
            t_years  : float or array — time since deployment [years]
            T_cell_K : float or array — cell temperature [K] (default from params)

        Returns
        -------
        dict with activity_Bq, P_beta_total_W, P_beta_absorbed_W,
                  Isc_uA, Voc_V, FF, P_out_W, P_out_uW,
                  eta_junction, fraction_remaining
        """
        t = inputs.get("t_years", 0.0)
        T = inputs.get("T_cell_K", None)

        return self._model.compute(t, T)

    def get_info(self) -> dict:
        return {
            "name": "Betavoltaic Cell",
            "ec_id": "EC222",
            "fidelity": "F1b",
            "description": (
                "P-N junction electrical model for betavoltaic cell: "
                "Isc(t) proportional to activity decay, Voc(T) with temperature "
                "coefficient and logarithmic activity correction, fill factor with "
                "slow radiation damage degradation. P_out = Isc * Voc * FF."
            ),
            "inputs": {
                "t_years": {"unit": "years", "range": [0, 500], "default": 0.0},
                "T_cell_K": {"unit": "K", "range": [200, 500], "default": 300.0},
            },
            "outputs": {
                "activity_Bq": {"unit": "Bq"},
                "P_beta_absorbed_W": {"unit": "W"},
                "Isc_uA": {"unit": "uA"},
                "Voc_V": {"unit": "V"},
                "FF": {"unit": "-"},
                "P_out_W": {"unit": "W"},
                "P_out_uW": {"unit": "uW"},
                "eta_junction": {"unit": "-"},
                "fraction_remaining": {"unit": "-"},
            },
            "source": "Olsen (1993); Sychov (2008); Prelas (2014); Sun (2018)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("Betavoltaic F1b — Design Point (t=0, T=300K):")
    r = model.predict({"t_years": 0.0, "T_cell_K": 300.0})
    for k, v in r.items():
        val = float(np.atleast_1d(v)[0])
        print(f"  {k:30s} = {val:.4g}")
    print("\nAfter 50 years:")
    r50 = model.predict({"t_years": 50.0, "T_cell_K": 300.0})
    for k, v in r50.items():
        val = float(np.atleast_1d(v)[0])
        print(f"  {k:30s} = {val:.4g}")
