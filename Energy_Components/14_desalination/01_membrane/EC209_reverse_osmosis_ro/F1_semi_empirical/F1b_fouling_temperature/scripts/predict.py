"""EC209 — Reverse Osmosis — F1b Fouling Temperature — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import ROF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ROF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict RO performance with fouling and temperature effects.

        Parameters
        ----------
        inputs : dict
            feed_salinity_ppm     : float (ppm, default 35000)
            feed_pressure_bar     : float (bar, default 60)
            feed_temperature_degC : float (degC, default 25)
            recovery_ratio        : float (0.3-0.6, default 0.45)
            operating_hours       : float (hours, default 0)
        """
        S = inputs.get("feed_salinity_ppm", 35000.0)
        P = inputs.get("feed_pressure_bar", 60.0)
        T = inputs.get("feed_temperature_degC", 25.0)
        r = inputs.get("recovery_ratio", 0.45)
        hours = np.asarray(inputs.get("operating_hours", 0.0), dtype=float)

        return self._model.compute(S, P, T, r, hours)

    def get_info(self) -> dict:
        return {
            "name": "Reverse Osmosis (RO)",
            "ec_id": "EC209",
            "fidelity": "F1b",
            "description": (
                "RO membrane model with fouling: A(t)=A0*exp(-k_foul*t/8760). "
                "Temperature: J(T)=J_ref*exp(2500*(1/T_ref-1/T)). "
                "Osmotic pressure pi=0.7*S/1000 bar."
            ),
            "inputs": {
                "feed_salinity_ppm": {"unit": "ppm", "range": [1000, 45000], "default": 35000},
                "feed_pressure_bar": {"unit": "bar", "range": [20, 80], "default": 60},
                "feed_temperature_degC": {"unit": "degC", "range": [10, 40], "default": 25},
                "recovery_ratio": {"unit": "dimensionless", "range": [0.3, 0.6], "default": 0.45},
                "operating_hours": {"unit": "hours", "range": [0, 87600], "default": 0},
            },
            "outputs": {
                "permeate_flow_m3_h": {"unit": "m3/h"},
                "sec_kwh_m3": {"unit": "kWh/m3"},
                "rejection_pct": {"unit": "%"},
                "flux_decline_factor": {"unit": "dimensionless"},
            },
            "source": "Elimelech & Phillip (2011); Kang & Cao (2012)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"feed_salinity_ppm": 35000, "feed_pressure_bar": 60,
                        "feed_temperature_degC": 25, "recovery_ratio": 0.45,
                        "operating_hours": 0})
    print("Design point (clean membrane, 25 degC):")
    for k, v in r.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")
