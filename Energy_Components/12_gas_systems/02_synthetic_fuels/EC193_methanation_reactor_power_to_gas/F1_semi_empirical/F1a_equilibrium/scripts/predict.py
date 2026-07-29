"""EC193 — Methanation Reactor — F1a Sabatier Equilibrium — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import MethanationF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MethanationF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict methanation reactor performance.

        Parameters
        ----------
        inputs : dict
            temperature   : float or array  (degC, 200–500)
            pressure      : float or array  (bar, 1–30)
            h2_co2_ratio  : float or array  (mol/mol, 3.5–5.0), default 4.0
            n_co2_in      : float  (mol/s), default 1.0

        Returns
        -------
        dict
            conversion         : CO2-to-CH4 conversion fraction (-)
            ch4_rate_mols      : CH4 production rate (mol/s)
            efficiency         : energy efficiency = X * LHV_CH4 / (4*LHV_H2)
            heat_released_kw   : exothermic reaction heat (kW)
        """
        T   = np.asarray(inputs["temperature"],   dtype=float)
        P   = np.asarray(inputs["pressure"],       dtype=float)
        r   = np.asarray(inputs.get("h2_co2_ratio", 4.0), dtype=float)
        n   = inputs.get("n_co2_in", None)

        X    = self._model.conversion(T, P, r)
        ch4  = self._model.ch4_rate(T, P, r, n)
        eta  = self._model.efficiency(T, P, r)
        Q    = self._model.heat_released(T, P, r, n)

        return {
            "conversion":       X,
            "ch4_rate_mols":    ch4,
            "efficiency":       eta,
            "heat_released_kw": Q,
        }

    def get_info(self) -> dict:
        return {
            "name": "Methanation Reactor (Power-to-Gas)",
            "ec_id": "EC193",
            "fidelity": "F1a",
            "description": (
                "Sabatier equilibrium: CO2 + 4H2 -> CH4 + 2H2O. "
                "X = X_max * exp(-k*(T-T_opt)^2/T_opt^2) * (P/P_ref)^0.1"
            ),
            "inputs": {
                "temperature":   {"unit": "degC",    "range": [200.0, 500.0]},
                "pressure":      {"unit": "bar",     "range": [1.0, 30.0]},
                "h2_co2_ratio":  {"unit": "mol/mol", "range": [3.5, 5.0], "default": 4.0},
                "n_co2_in":      {"unit": "mol/s",   "default": 1.0},
            },
            "outputs": {
                "conversion":       {"unit": "dimensionless"},
                "ch4_rate_mols":    {"unit": "mol/s"},
                "efficiency":       {"unit": "dimensionless"},
                "heat_released_kw": {"unit": "kW"},
            },
            "source": "Gao et al. (2012), RSC Advances, 2, 2358-2368",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"temperature": 300.0, "pressure": 10.0, "h2_co2_ratio": 4.0})
    print(f"T=300°C, P=10 bar, H2/CO2=4:")
    print(f"  Conversion       = {float(r['conversion']):.4f}")
    print(f"  CH4 rate         = {float(r['ch4_rate_mols']):.4f} mol/s")
    print(f"  Efficiency       = {float(r['efficiency']):.4f} ({float(r['efficiency'])*100:.2f}%)")
    print(f"  Heat released    = {float(r['heat_released_kw']):.2f} kW")
