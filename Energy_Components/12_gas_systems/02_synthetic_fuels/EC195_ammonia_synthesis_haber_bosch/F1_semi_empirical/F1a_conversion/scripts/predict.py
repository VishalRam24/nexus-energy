"""EC195 — Ammonia Synthesis (Haber-Bosch) — F1a Conversion Model — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import AmmoniaHaberBoschF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AmmoniaHaberBoschF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict Haber-Bosch ammonia synthesis performance.

        Parameters
        ----------
        inputs : dict
            temperature : float or array  (degC, 350–550)
            pressure    : float or array  (bar, 100–300)
            n_n2_in     : float  (mol/s), default 1.0

        Returns
        -------
        dict
            conversion_per_pass : per-pass N2 conversion (-)
            nh3_rate_kgs        : NH3 production rate (kg/s)
            energy_gj_per_ton   : specific energy consumption (GJ/tNH3)
            efficiency          : energy efficiency (-)
        """
        T   = np.asarray(inputs["temperature"], dtype=float)
        P   = np.asarray(inputs["pressure"],    dtype=float)
        n   = inputs.get("n_n2_in", None)

        X    = self._model.per_pass_conversion(T, P)
        rate = self._model.nh3_rate(T, P, n)
        E    = self._model.energy_per_ton(T, P)
        eta  = self._model.efficiency(T, P)

        return {
            "conversion_per_pass": X,
            "nh3_rate_kgs":        rate,
            "energy_gj_per_ton":   E,
            "efficiency":          eta,
        }

    def get_info(self) -> dict:
        return {
            "name": "Ammonia Synthesis (Haber-Bosch)",
            "ec_id": "EC195",
            "fidelity": "F1a",
            "description": (
                "N2 + 3H2 → 2NH3. Per-pass conversion: "
                "X = min(X_ref*(P/P_ref)^0.5*exp(-Ea/R*(1/T-1/T_ref)), X_eq(T,P))"
            ),
            "inputs": {
                "temperature": {"unit": "degC", "range": [350.0, 550.0]},
                "pressure":    {"unit": "bar",  "range": [100.0, 300.0]},
                "n_n2_in":     {"unit": "mol/s", "default": 1.0},
            },
            "outputs": {
                "conversion_per_pass": {"unit": "dimensionless"},
                "nh3_rate_kgs":        {"unit": "kg/s"},
                "energy_gj_per_ton":   {"unit": "GJ/tNH3"},
                "efficiency":          {"unit": "dimensionless"},
            },
            "source": "Appl (2011), Ullmann's Encyclopedia of Industrial Chemistry",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"temperature": 450.0, "pressure": 200.0})
    print(f"T=450°C, P=200 bar:")
    print(f"  Conversion per pass = {float(r['conversion_per_pass']):.4f}")
    print(f"  NH3 rate            = {float(r['nh3_rate_kgs'])*1000:.4f} g/s")
    print(f"  Specific energy     = {float(r['energy_gj_per_ton']):.2f} GJ/tNH3")
    print(f"  Efficiency          = {float(r['efficiency']):.4f}")
