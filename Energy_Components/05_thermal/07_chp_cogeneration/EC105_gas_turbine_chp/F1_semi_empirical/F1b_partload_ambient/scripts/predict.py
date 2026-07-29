"""EC105 -- Gas Turbine CHP -- F1b Part-Load + Ambient + HRSG -- Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import GasTurbineCHPF1b


class ComponentModel:
    """Standardized interface for EC105 Gas Turbine CHP -- F1b model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = GasTurbineCHPF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            PLR          : float or array [0.4 - 1.0]
            T_ambient    : float or array [K] (default 288.15 = 15 degC ISO)
            P_ambient    : float or array [kPa] (default 101.325)
        returns:
            efficiency_electrical : electrical efficiency [-]
            efficiency_thermal    : HRSG thermal efficiency [-]
            efficiency_total      : total first-law CHP efficiency [-]
            power_electrical_kw   : electrical output [kW_e]
            heat_recovery_kw      : HRSG heat recovery [kW_th]
            fuel_input_kw         : fuel input [kW_fuel, LHV]
            exhaust_temp_K        : HRSG inlet exhaust temperature [K]
            heat_to_power_ratio   : HPR = Q_th / P_el [-]
            heat_rate_kj_kwh      : electrical heat rate [kJ/kWh]
        """
        PLR   = np.asarray(inputs["PLR"], dtype=float)
        T_amb = np.asarray(inputs.get("T_ambient", 288.15), dtype=float)
        P_amb = np.asarray(inputs.get("P_ambient", 101.325), dtype=float)

        return {
            "efficiency_electrical": self._model.eta_electrical(PLR, T_amb, P_amb),
            "efficiency_thermal":    self._model.eta_thermal(PLR, T_amb),
            "efficiency_total":      self._model.eta_total(PLR, T_amb, P_amb),
            "power_electrical_kw":   self._model.power_electrical_kw(PLR, T_amb, P_amb),
            "heat_recovery_kw":      self._model.heat_recovery_kw(PLR, T_amb, P_amb),
            "fuel_input_kw":         self._model.fuel_input_kw(PLR, T_amb, P_amb),
            "exhaust_temp_K":        self._model.exhaust_temp_k(PLR),
            "heat_to_power_ratio":   self._model.heat_to_power_ratio(PLR, T_amb, P_amb),
            "heat_rate_kj_kwh":      self._model.heat_rate_kj_kwh(PLR, T_amb, P_amb),
        }

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Gas Turbine CHP",
            "ec_id": "EC105",
            "fidelity": "F1b",
            "model": "Part-Load + Ambient Temperature + HRSG Exhaust Temperature",
            "description": (
                "eta_el(PLR,T,P) = eta_el_rated * (a+b*PLR+c*PLR^2) * sqrt(T_ref/T_amb) * (P/P_ref); "
                "T_exh(PLR) = T_exh_rated + dT*(1-PLR); "
                "eta_th(PLR,T_exh) = eta_th_rated * (th_a+th_b*PLR) * (1+coeff*(T_exh-T_ref))"
            ),
            "inputs": {
                "PLR":       {"unit": "-", "range": [0.4, 1.0]},
                "T_ambient": {"unit": "K", "range": [248.15, 323.15], "default": 288.15},
                "P_ambient": {"unit": "kPa", "range": [85.0, 107.0], "default": 101.325},
            },
            "outputs": {
                "efficiency_electrical": {"unit": "-"},
                "efficiency_thermal":    {"unit": "-"},
                "efficiency_total":      {"unit": "-"},
                "power_electrical_kw":   {"unit": "kW_e"},
                "heat_recovery_kw":      {"unit": "kW_th"},
                "fuel_input_kw":         {"unit": "kW_fuel"},
                "exhaust_temp_K":        {"unit": "K"},
                "heat_to_power_ratio":   {"unit": "-"},
                "heat_rate_kj_kwh":      {"unit": "kJ/kWh"},
            },
            "source": "US EPA CHP Catalog (2017); Kehlhofer et al. (2009); Walsh & Fletcher (2004); ISO 2314:2009",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC105 F1b -- ISO conditions (PLR=1.0, 15C, 101.325 kPa):")
    r = model.predict({"PLR": 1.0})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
    print("\nPart-load, hot day (PLR=0.6, 40C=313.15K, 101.325 kPa):")
    r = model.predict({"PLR": 0.6, "T_ambient": 313.15})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
