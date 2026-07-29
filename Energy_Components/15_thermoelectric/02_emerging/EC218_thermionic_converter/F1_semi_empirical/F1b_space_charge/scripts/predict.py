"""EC218 — Thermionic Converter — F1b Space-Charge — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import ThermionicF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ThermionicF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict thermionic converter performance with space-charge correction.

        Parameters
        ----------
        inputs : dict
            T_emitter_K  : float or array (K, 1200-2200)
            T_collector_K: float or array (K, 400-1200)

        Returns
        -------
        dict with phi_e_eV, phi_c_eV, J_emitter_Am2, J_collector_Am2, J_net_Am2,
                  V_open_V, V_terminal_V, power_w, power_density_w_cm2, heat_input_w, efficiency
        """
        T_e = inputs.get("T_emitter_K", 1700.0)
        T_c = inputs.get("T_collector_K", 900.0)
        return self._model.compute(T_e, T_c)

    def get_info(self) -> dict:
        return {
            "name": "Thermionic Converter",
            "ec_id": "EC218",
            "fidelity": "F1b",
            "description": (
                "Richardson-Dushman thermionic emission with: "
                "(1) temperature-dependent work functions phi(T) = phi0 + dphi/dT*(T-T0); "
                "(2) space-charge correction factor for Cs-vapor diode; "
                "(3) back-emission from collector; "
                "(4) lead resistance voltage drop."
            ),
            "inputs": {
                "T_emitter_K": {"unit": "K", "range": [1200, 2200], "default": 1700.0},
                "T_collector_K": {"unit": "K", "range": [400, 1200], "default": 900.0},
            },
            "outputs": {
                "phi_e_eV": {"unit": "eV"},
                "phi_c_eV": {"unit": "eV"},
                "J_emitter_Am2": {"unit": "A/m^2"},
                "J_collector_Am2": {"unit": "A/m^2"},
                "J_net_Am2": {"unit": "A/m^2"},
                "V_open_V": {"unit": "V"},
                "V_terminal_V": {"unit": "V"},
                "power_w": {"unit": "W"},
                "power_density_w_cm2": {"unit": "W/cm^2"},
                "heat_input_w": {"unit": "W"},
                "efficiency": {"unit": "-"},
            },
            "source": "Hatsopoulos & Gyftopoulos (1979); Houston (1959); Rasor (1991)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_emitter_K": 1700.0, "T_collector_K": 900.0})
    print("Design point (T_emitter=1700K, T_collector=900K):")
    for k, v in r.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4e}")
