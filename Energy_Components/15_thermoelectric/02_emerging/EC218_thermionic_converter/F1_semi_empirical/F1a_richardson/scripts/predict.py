"""EC218 — Thermionic Converter — F1a Richardson Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import ThermionicF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ThermionicF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            T_emitter   : float or array — emitter temperature [K]
            T_collector : float or array — collector temperature [K]
        returns:
            J_emitter_Am2  : A/m^2
            J_collector_Am2: A/m^2
            J_net_Am2      : A/m^2
            V_out_V        : V
            power_w        : W
            heat_input_w   : W
            efficiency     : dimensionless
        """
        T_e = np.asarray(inputs["T_emitter"], dtype=float)
        T_c = np.asarray(inputs["T_collector"], dtype=float)
        return self._model.compute(T_e, T_c)

    def get_info(self) -> dict:
        return {
            "name": "Thermionic Converter",
            "ec_id": "EC218",
            "fidelity": "F1a",
            "description": "J = A*T^2*exp(-phi/(k_B*T)); V_out = phi_e - phi_c; eta = P/Q_in",
            "inputs": {
                "T_emitter":   {"unit": "K", "range": [1200.0, 2000.0]},
                "T_collector": {"unit": "K", "range": [400.0, 1200.0]},
            },
            "outputs": {
                "J_emitter_Am2":   {"unit": "A/m^2"},
                "J_collector_Am2": {"unit": "A/m^2"},
                "J_net_Am2":       {"unit": "A/m^2"},
                "V_out_V":         {"unit": "V"},
                "power_w":         {"unit": "W"},
                "heat_input_w":    {"unit": "W"},
                "efficiency":      {"unit": "-"},
            },
            "source": "Hatsopoulos & Gyftopoulos (1979); Angrist (1982) Direct Energy Conversion",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_emitter": 1700.0, "T_collector": 900.0})
    print(f"Thermionic at 1700/900K: eta={float(r['efficiency'])*100:.1f}%, "
          f"J_net={float(r['J_net_Am2']):.1f} A/m^2, "
          f"P={float(r['power_w'])*1000:.2f} mW, "
          f"V={float(r['V_out_V']):.2f} V")
