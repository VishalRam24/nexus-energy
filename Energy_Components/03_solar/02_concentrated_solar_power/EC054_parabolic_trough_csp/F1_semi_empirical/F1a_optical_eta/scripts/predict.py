"""EC054 — Parabolic Trough CSP — F1a Optical Efficiency — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import ParabolicTroughF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ParabolicTroughF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
          dni             : Direct Normal Irradiance (W/m2)
          T_absorber      : absorber / HTF temperature (degC)
          T_ambient       : ambient temperature (degC)
          incidence_angle : angle of incidence on collector aperture (deg)
        returns:
          useful_heat_kw, optical_efficiency, thermal_loss_kw, overall_efficiency
        """
        dni = np.asarray(inputs["dni"], dtype=float)
        T_abs = np.asarray(inputs["T_absorber"], dtype=float)
        T_amb = np.asarray(inputs["T_ambient"], dtype=float)
        theta = np.asarray(inputs["incidence_angle"], dtype=float)

        return self._model.predict_all(dni, T_abs, T_amb, theta)

    def get_info(self) -> dict:
        return {
            "name": "Parabolic Trough CSP",
            "ec_id": "EC054",
            "fidelity": "F1a",
            "description": "Q_useful = DNI*A*eta_opt*IAM(theta) - Q_loss(T_abs,T_amb); Schott PTR70 receiver",
            "inputs": {
                "dni":             {"unit": "W/m2", "range": [0.0, 1000.0]},
                "T_absorber":      {"unit": "degC", "range": [100.0, 400.0]},
                "T_ambient":       {"unit": "degC", "range": [0.0, 50.0]},
                "incidence_angle": {"unit": "deg",  "range": [0.0, 80.0]},
            },
            "outputs": {
                "useful_heat_kw":    {"unit": "kW"},
                "optical_efficiency":{"unit": "dimensionless"},
                "thermal_loss_kw":   {"unit": "kW"},
                "overall_efficiency":{"unit": "dimensionless"},
            },
            "source": "Forristall (2003), NREL/TP-550-34169",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"dni": 850.0, "T_absorber": 300.0, "T_ambient": 25.0, "incidence_angle": 0.0})
    print(f"DNI=850, T_abs=300C, theta=0: "
          f"Q_useful={float(r['useful_heat_kw']):.1f}kW, "
          f"eta_opt={float(r['optical_efficiency']):.3f}, "
          f"Q_loss={float(r['thermal_loss_kw']):.2f}kW, "
          f"eta_overall={float(r['overall_efficiency']):.3f}")
