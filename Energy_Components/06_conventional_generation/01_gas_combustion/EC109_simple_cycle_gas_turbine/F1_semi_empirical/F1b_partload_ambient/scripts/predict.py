"""EC109 -- Simple Cycle Gas Turbine -- F1b Part-Load + Ambient -- Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import SimpleCycleGasTurbineF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SimpleCycleGasTurbineF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            PLR         : float or array [0.3 - 1.0]
            T_ambient   : float or array [K] (default 288.15)
            P_ambient   : float or array [kPa] (default 101.325)
            fuel_lhv    : float [MJ/kg] (default 50.0, optional override)
        returns:
            efficiency          : net LHV electrical efficiency [-]
            power_output_kw     : electrical output [kW]
            fuel_flow_kg_s      : fuel mass flow [kg/s]
            exhaust_temp_K      : exhaust temperature [K]
            heat_rate_kj_kwh    : heat rate [kJ/kWh]
        """
        PLR   = np.asarray(inputs["PLR"], dtype=float)
        T_amb = np.asarray(inputs.get("T_ambient", 288.15), dtype=float)
        P_amb = np.asarray(inputs.get("P_ambient", 101.325), dtype=float)

        # Optional: override fuel LHV
        if "fuel_lhv" in inputs:
            self._model.LHV = float(inputs["fuel_lhv"])

        return {
            "efficiency":       self._model.efficiency(PLR, T_amb, P_amb),
            "power_output_kw":  self._model.power_output_kw(PLR, T_amb, P_amb),
            "fuel_flow_kg_s":   self._model.fuel_flow_kg_s(PLR, T_amb, P_amb),
            "exhaust_temp_K":   self._model.exhaust_temp_k(PLR),
            "heat_rate_kj_kwh": self._model.heat_rate_kj_kwh(PLR, T_amb, P_amb),
        }

    def get_info(self) -> dict:
        return {
            "name": "Simple Cycle Gas Turbine",
            "ec_id": "EC109",
            "fidelity": "F1b",
            "model": "Part-Load + Ambient Correction",
            "description": (
                "eta = eta_rated * (a + b*PLR + c*PLR^2) * sqrt(T_ref/T_amb); "
                "P = P_rated * PLR * (P_amb/P_ref) * sqrt(T_ref/T_amb)"
            ),
            "inputs": {
                "PLR":       {"unit": "-", "range": [0.3, 1.0]},
                "T_ambient": {"unit": "K", "range": [243.15, 323.15], "default": 288.15},
                "P_ambient": {"unit": "kPa", "range": [80.0, 110.0], "default": 101.325},
                "fuel_lhv":  {"unit": "MJ/kg", "range": [40.0, 55.0], "default": 50.0},
            },
            "outputs": {
                "efficiency":       {"unit": "-"},
                "power_output_kw":  {"unit": "kW"},
                "fuel_flow_kg_s":   {"unit": "kg/s"},
                "exhaust_temp_K":   {"unit": "K"},
                "heat_rate_kj_kwh": {"unit": "kJ/kWh"},
            },
            "source": "Walsh & Fletcher (2004); ISO 2314:2009",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC109 F1b -- ISO conditions (PLR=1.0, T=288.15K, P=101.325kPa):")
    r = model.predict({"PLR": 1.0, "T_ambient": 288.15, "P_ambient": 101.325})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
    print("\nHot day (PLR=0.8, T=313.15K / 40C, P=101.325kPa):")
    r = model.predict({"PLR": 0.8, "T_ambient": 313.15, "P_ambient": 101.325})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
