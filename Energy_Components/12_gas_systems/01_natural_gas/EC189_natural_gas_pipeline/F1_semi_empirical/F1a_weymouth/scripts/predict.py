"""EC189 — Natural Gas Pipeline — F1a Weymouth — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import NGPipelineF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = NGPipelineF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict natural gas pipeline flow and pressure drop.

        Args:
            inputs: dict with keys:
                - length_km   : pipeline length [km]
                - diameter_m  : internal pipe diameter [m]
                - P_in_bar    : inlet pressure [bar]
                - P_out_bar   : outlet pressure [bar]
                - T_K         : average gas temperature [K], optional, default 288.15
                - Z           : compressibility factor [-], optional, default 0.9
                - E           : efficiency factor [-], optional, default 0.92

        Returns:
            dict with keys:
                - Q_std_m3_per_day  : gas flow at standard conditions [m³/day]
                - Q_std_m3_per_s    : gas flow at standard conditions [m³/s]
                - Q_kg_per_s        : mass flow [kg/s]
                - pressure_drop_bar : P_in - P_out [bar]
                - weymouth_f        : Weymouth friction factor [-]
        """
        L = np.asarray(inputs["length_km"], dtype=float)
        D = np.asarray(inputs["diameter_m"], dtype=float)
        P1 = np.asarray(inputs["P_in_bar"], dtype=float)
        P2 = np.asarray(inputs["P_out_bar"], dtype=float)
        T = np.asarray(inputs.get("T_K", 288.15), dtype=float)
        Z = np.asarray(inputs.get("Z", 0.9), dtype=float)
        E = np.asarray(inputs.get("E", 0.92), dtype=float)

        return {
            "Q_std_m3_per_day":  self._model.flow_rate_std_m3_per_day(L, D, P1, P2, T, Z, E),
            "Q_std_m3_per_s":    self._model.flow_rate_std_m3_per_s(L, D, P1, P2, T, Z, E),
            "Q_kg_per_s":        self._model.flow_rate_kg_per_s(L, D, P1, P2, T, Z, E),
            "pressure_drop_bar": P1 - P2,
            "weymouth_f":        self._model.weymouth_friction_factor(D),
        }

    def get_info(self) -> dict:
        return {
            "name": "Natural Gas Pipeline",
            "ec_id": "EC189",
            "fidelity": "F1a",
            "description": "Weymouth isothermal steady-state gas pipeline flow equation",
            "inputs": {
                "length_km":  {"unit": "km",  "range": [1.0, 2000.0]},
                "diameter_m": {"unit": "m",   "range": [0.1, 1.5]},
                "P_in_bar":   {"unit": "bar", "range": [5.0, 150.0]},
                "P_out_bar":  {"unit": "bar", "range": [1.0, 140.0]},
                "T_K":        {"unit": "K",   "range": [250.0, 320.0], "default": 288.15},
                "Z":          {"unit": "-",   "range": [0.7, 1.0],     "default": 0.9},
                "E":          {"unit": "-",   "range": [0.85, 0.95],   "default": 0.92},
            },
            "outputs": {
                "Q_std_m3_per_day": {"unit": "m³/day"},
                "Q_std_m3_per_s":   {"unit": "m³/s"},
                "Q_kg_per_s":       {"unit": "kg/s"},
                "pressure_drop_bar": {"unit": "bar"},
                "weymouth_f":       {"unit": "dimensionless"},
            },
            "source": "Weymouth (1912); Menon (2005) Gas Pipeline Hydraulics",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for P_out in [30, 40, 50, 60, 70]:
        r = model.predict({"length_km": 100.0, "diameter_m": 0.6096,
                           "P_in_bar": 75.0, "P_out_bar": float(P_out)})
        print(f"P_out={P_out} bar: Q={float(r['Q_std_m3_per_day'])/1e6:.3f} Mm³/day, "
              f"dP={float(r['pressure_drop_bar']):.1f} bar")
