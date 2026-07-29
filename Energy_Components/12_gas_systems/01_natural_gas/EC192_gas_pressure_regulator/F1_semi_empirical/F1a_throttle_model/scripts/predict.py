"""EC192 — Gas Pressure Regulator — F1a Throttle Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import GasPressureRegulatorF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = GasPressureRegulatorF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict gas pressure regulator (PRV/PCV) performance.

        Args:
            inputs: dict with keys:
                - P_up_bar    : upstream pressure [bar]
                - P_down_bar  : downstream (set) pressure [bar]  must be ≤ P_up
                - T_up_K      : upstream temperature [K], optional, default 288.15
                - Z           : compressibility factor [-], optional, default 0.9
                - Cv          : valve flow coefficient [-], optional, default 500

        Returns:
            dict with keys:
                - T_down_K        : downstream temperature after JT cooling [K]
                - delta_T_K       : temperature change [K] (negative = cooling)
                - Q_std_m3_per_h  : gas flow [m³/h at std]
                - Q_kg_per_s      : mass flow [kg/s]
                - expansion_Y     : ISA expansion factor [-]
                - is_choked       : choked flow flag [bool]
        """
        P_up = np.asarray(inputs["P_up_bar"], dtype=float)
        P_down = np.asarray(inputs["P_down_bar"], dtype=float)
        T_up = np.asarray(inputs.get("T_up_K", 288.15), dtype=float)
        Z = np.asarray(inputs.get("Z", 0.9), dtype=float)
        Cv = inputs.get("Cv", None)

        T_down = self._model.temperature_out(T_up, P_up, P_down)

        return {
            "T_down_K":       T_down,
            "delta_T_K":      T_down - T_up,
            "Q_std_m3_per_h": self._model.flow_std_m3_per_h(P_up, P_down, T_up, Z, Cv),
            "Q_kg_per_s":     self._model.flow_kg_per_s(P_up, P_down, T_up, Z, Cv),
            "expansion_Y":    self._model.expansion_factor_Y(P_up, P_down),
            "is_choked":      self._model.is_choked(P_up, P_down),
        }

    def get_info(self) -> dict:
        return {
            "name": "Gas Pressure Regulator",
            "ec_id": "EC192",
            "fidelity": "F1a",
            "description": "Isenthalpic throttle (JT cooling) with ISA Cv gas flow equation",
            "inputs": {
                "P_up_bar":   {"unit": "bar", "range": [5.0, 200.0]},
                "P_down_bar": {"unit": "bar", "range": [1.0, 199.0]},
                "T_up_K":     {"unit": "K",   "range": [250.0, 400.0], "default": 288.15},
                "Z":          {"unit": "-",   "range": [0.7, 1.0],     "default": 0.9},
                "Cv":         {"unit": "gal/min/psi^0.5", "range": [10.0, 5000.0], "default": 500.0},
            },
            "outputs": {
                "T_down_K":       {"unit": "K"},
                "delta_T_K":      {"unit": "K", "note": "negative = cooling"},
                "Q_std_m3_per_h": {"unit": "m³/h"},
                "Q_kg_per_s":     {"unit": "kg/s"},
                "expansion_Y":    {"unit": "dimensionless"},
                "is_choked":      {"unit": "bool"},
            },
            "source": "ANSI/ISA-75.01.01-2012; Burnett (1999)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(f"{'P_down':>8} {'T_down':>10} {'ΔT':>8} {'Q_m3h':>12} {'Choked':>8}")
    for P_down in [70, 50, 30, 10, 2]:
        r = model.predict({"P_up_bar": 80.0, "P_down_bar": float(P_down),
                           "T_up_K": 288.15})
        print(f"{P_down:>8} {float(r['T_down_K']):>10.2f} {float(r['delta_T_K']):>8.2f} "
              f"{float(r['Q_std_m3_per_h']):>12.1f} {bool(r['is_choked']):>8}")
