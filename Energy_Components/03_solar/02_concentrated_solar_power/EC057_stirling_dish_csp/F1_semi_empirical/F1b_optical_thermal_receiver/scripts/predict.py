"""EC057 — Stirling Dish CSP — F1b Optical+Receiver Thermal Loss — Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import StirlingDishF1b


class ComponentModel:
    """Standardized interface for EC057 Stirling Dish CSP — F1b model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = StirlingDishF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {
                "dni_w_m2"       : float or array [W/m2]
                "theta_deg"      : float or array [deg] — tracking residual angle
                "T_receiver_degC": float or array [degC] — receiver cavity temperature
                "T_ambient_degC" : float or array [degC]
                "PLR"            : float or array [0.3-1.0] (optional, default 1.0)
            }
        Returns:
            {
                "power_output_kw"    : net electrical output [kW],
                "Q_absorbed_kw"      : solar power absorbed by receiver [kW],
                "Q_receiver_loss_kw" : cavity heat loss [kW],
                "Q_net_thermal_kw"   : net thermal to Stirling engine [kW],
                "eta_stirling"       : Stirling thermal-to-electrical efficiency [-],
                "overall_efficiency" : system efficiency (P_elec / DNI*A_dish) [-],
                "iam_factor"         : incidence angle modifier [-]
            }
        """
        dni      = np.asarray(inputs.get("dni_w_m2", 0.0), dtype=float)
        theta    = np.asarray(inputs.get("theta_deg", 0.0), dtype=float)
        T_rec    = np.asarray(inputs.get("T_receiver_degC", self._model.T_rec_design), dtype=float)
        T_amb    = np.asarray(inputs.get("T_ambient_degC", 20.0), dtype=float)
        PLR      = np.asarray(inputs.get("PLR", 1.0), dtype=float)
        return self._model.predict_all(dni, theta, T_rec, T_amb, PLR)

    def get_info(self) -> dict:
        return {
            "name": "Stirling Dish CSP",
            "ec_id": "EC057",
            "fidelity": "F1b",
            "model": "Optical efficiency + receiver cavity heat loss + Stirling engine",
            "description": (
                "P_elec = (Q_abs - Q_loss_cav) * eta_Stirling * f_PLR; "
                "Receiver losses include convection, conduction, radiation; "
                "Stirling efficiency = eta_int * Carnot(T_hot, T_sink)"
            ),
            "inputs": {
                "dni_w_m2":        {"unit": "W/m2", "range": [0, 1100]},
                "theta_deg":       {"unit": "deg",  "range": [0, 10], "note": "tracking error"},
                "T_receiver_degC": {"unit": "degC", "range": [400, 800]},
                "T_ambient_degC":  {"unit": "degC", "range": [-10, 45]},
                "PLR":             {"unit": "-",    "range": [0.3, 1.0]},
            },
            "outputs": {
                "power_output_kw":    {"unit": "kW"},
                "Q_absorbed_kw":      {"unit": "kW"},
                "Q_receiver_loss_kw": {"unit": "kW"},
                "Q_net_thermal_kw":   {"unit": "kW"},
                "eta_stirling":       {"unit": "-"},
                "overall_efficiency": {"unit": "-"},
                "iam_factor":         {"unit": "-"},
            },
            "source": "Mancini et al. (2003); Nepveu et al. (2009); Stine & Diver (1994)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC057 F1b — Stirling Dish CSP — Design conditions:")
    r = model.predict({"dni_w_m2": 900.0, "theta_deg": 0.0, "T_receiver_degC": 720.0,
                        "T_ambient_degC": 25.0, "PLR": 1.0})
    for k, v in r.items():
        print(f"  {k}: {float(np.atleast_1d(v)[0]):.4f}")
