"""EC051 — DSSC — F1b Single-Diode + Thermal — Standardized Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import DSSCf1b


class ComponentModel:
    """Standardized interface for EC051 DSSC — F1b single-diode + Faiman NOCT + DSSC correction."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = DSSCf1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "irradiance_w_m2": W/m2 (0-1200),
                "T_ambient_degC": degC (-10 to 60)
            }
        Returns:
            i_mp, v_mp, p_mp, i_sc, v_oc, fill_factor, efficiency, T_cell_c
        """
        G = np.asarray(inputs["irradiance_w_m2"], dtype=float)
        T = np.asarray(inputs["T_ambient_degC"], dtype=float)
        result = self._model.mpp(G, T)
        result["efficiency"] = self._model.efficiency(G, T)
        return result

    def get_info(self) -> dict:
        return {
            "name": "Dye-Sensitized Solar Cell (DSSC)",
            "ec_id": "EC051",
            "fidelity": "F1b",
            "description": (
                "De Soto 5-parameter single-diode + Faiman NOCT thermal model + empirical "
                "DSSC tempco correction (factor 0.60). Accounts for TiO2/electrolyte "
                "recombination barrier weaker T-dependence vs semiconductor I_o scaling."
            ),
            "inputs": {
                "irradiance_w_m2": {"unit": "W/m2", "range": [0.0, 1200.0]},
                "T_ambient_degC": {"unit": "degC", "range": [-10.0, 60.0]},
            },
            "outputs": {
                "i_mp": {"unit": "A"}, "v_mp": {"unit": "V"}, "p_mp": {"unit": "W"},
                "i_sc": {"unit": "A"}, "v_oc": {"unit": "V"},
                "fill_factor": {"unit": "dimensionless"},
                "efficiency": {"unit": "dimensionless"},
                "T_cell_c": {"unit": "degC"},
            },
            "source": "Snaith & Grätzel (2007); Toivola et al. (2009); Cameron et al. (2005)",
            "library": "NumPy/SciPy",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": 25.0})
    print("\nAt 1000 W/m2, T_amb=25C:")
    for k, v in r.items():
        print(f"  {k}: {float(v):.5f}")
