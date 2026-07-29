"""EC053 — TPV — F1b Two-Diode + Thermal — Standardized Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import TPVf1b


class ComponentModel:
    """Standardized interface for EC053 TPV — F1b two-diode + emitter thermal model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = TPVf1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "T_emitter_K": emitter temperature (K, 800-2000),
                "T_heatsink_degC": heat sink temperature (degC, optional, default 25)
            }
        Returns:
            i_mp, v_mp, p_mp, i_sc, v_oc, fill_factor, efficiency, T_cell_c
        """
        T_e = np.asarray(inputs["T_emitter_K"], dtype=float)
        T_hs = inputs.get("T_heatsink_degC", None)
        if T_hs is not None:
            T_hs = np.asarray(T_hs, dtype=float)
        result = self._model.mpp(T_e, T_hs)
        result["efficiency"] = self._model.efficiency(T_e, T_hs)
        return result

    def get_info(self) -> dict:
        return {
            "name": "Thermophotovoltaic (TPV)",
            "ec_id": "EC053",
            "fidelity": "F1b",
            "description": (
                "Two-diode model with emitter-temperature-dependent photocurrent "
                "using Wien approximation of above-bandgap Planck photon flux. "
                "Heat sink thermal resistance model for cell temperature. "
                "Representative of GaSb (Eg=0.72eV) TPV cells at ~1500K emitter."
            ),
            "inputs": {
                "T_emitter_K": {"unit": "K", "range": [800.0, 2000.0]},
                "T_heatsink_degC": {"unit": "degC", "range": [0.0, 60.0], "optional": True},
            },
            "outputs": {
                "i_mp": {"unit": "A"}, "v_mp": {"unit": "V"}, "p_mp": {"unit": "W"},
                "i_sc": {"unit": "A"}, "v_oc": {"unit": "V"},
                "fill_factor": {"unit": "dimensionless"},
                "efficiency": {"unit": "dimensionless", "note": "P_mp / (sigma*T^4*area)"},
                "T_cell_c": {"unit": "degC"},
            },
            "source": "Coutts (1999) RSER; Bauer (2011) Springer; Datas & Algora (2010)",
            "library": "NumPy/SciPy (brentq solver)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"T_emitter_K": 1500.0, "T_heatsink_degC": 25.0})
    print("\nAt T_emitter=1500K, T_heatsink=25C:")
    for k, v in r.items():
        print(f"  {k}: {float(v):.5f}")
