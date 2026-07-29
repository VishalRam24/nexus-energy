"""EC055 — Solar Tower CSP — F1a Optical+Thermal — Standardized Predict Interface"""

import json, numpy as np
from pathlib import Path
from model import SolarTowerF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SolarTowerF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
          dni            : Direct Normal Irradiance (W/m2)
          solar_zenith   : solar zenith angle (deg)  [0=overhead, 90=horizon]
          T_receiver     : receiver surface temperature (degC)
          T_ambient      : ambient temperature (degC)
        returns:
          Q_field_kw, useful_heat_kw, thermal_loss_kw,
          optical_efficiency, receiver_efficiency, overall_efficiency
        """
        dni = np.asarray(inputs["dni"], dtype=float)
        z = np.asarray(inputs["solar_zenith"], dtype=float)
        T_r = np.asarray(inputs["T_receiver"], dtype=float)
        T_a = np.asarray(inputs["T_ambient"], dtype=float)
        return self._model.predict_all(dni, z, T_r, T_a)

    def get_info(self) -> dict:
        return {
            "name": "Solar Tower Central Receiver CSP",
            "ec_id": "EC055",
            "fidelity": "F1a",
            "description": "Heliostat field optical efficiency * cos*atten - receiver radiative+convective losses",
            "inputs": {
                "dni":          {"unit": "W/m2", "range": [0.0, 1100.0]},
                "solar_zenith": {"unit": "deg",  "range": [0.0, 85.0]},
                "T_receiver":   {"unit": "degC", "range": [300.0, 800.0]},
                "T_ambient":    {"unit": "degC", "range": [-10.0, 50.0]},
            },
            "outputs": {
                "Q_field_kw":          {"unit": "kW"},
                "useful_heat_kw":      {"unit": "kW"},
                "thermal_loss_kw":     {"unit": "kW"},
                "optical_efficiency":  {"unit": "dimensionless"},
                "receiver_efficiency": {"unit": "dimensionless"},
                "overall_efficiency":  {"unit": "dimensionless"},
            },
            "source": "Wagner & Wendelin (2018) Solar Energy 171; Falcone (1986) SAND86-8009",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(json.dumps(model.get_info(), indent=2))
    r = model.predict({"dni": 900.0, "solar_zenith": 20.0,
                       "T_receiver": 565.0, "T_ambient": 25.0})
    print("\nDNI=900, zen=20, T_recv=565C, T_amb=25C:")
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
