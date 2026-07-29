"""EC190 — LNG Regasification Terminal — F1a Energy Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import LNGRegasF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = LNGRegasF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict LNG regasification terminal performance.

        Args:
            inputs: dict with keys:
                - sendout_rate_ton_per_h : LNG sendout rate [ton/h]
                - sec_kwh_per_ton        : specific energy consumption [kWh/ton], optional, default 50
                - T_ambient_K            : ambient temperature [K], optional, default 288.15
                - f_cold                 : cold energy recovery fraction [-], optional, default 0.0

        Returns:
            dict with keys:
                - power_demand_kw        : gross electrical/heat power demand [kW]
                - cold_recovery_kw       : cold energy recovered [kW]
                - net_power_kw           : net power consumption [kW]
                - gas_sendout_kg_per_s   : gas sendout mass flow [kg/s]
                - gas_sendout_m3_per_day : gas sendout at std conditions [m³/day]
                - net_sec_kwh_per_ton    : net SEC after cold recovery [kWh/ton]
        """
        m = np.asarray(inputs["sendout_rate_ton_per_h"], dtype=float)
        sec = np.asarray(inputs.get("sec_kwh_per_ton", 50.0), dtype=float)
        T_amb = np.asarray(inputs.get("T_ambient_K", 288.15), dtype=float)
        f_cold = np.asarray(inputs.get("f_cold", 0.0), dtype=float)

        return {
            "power_demand_kw":        self._model.power_demand_kw(m, sec),
            "cold_recovery_kw":       self._model.cold_energy_recovery_kw(m, T_amb, f_cold),
            "net_power_kw":           self._model.net_power_kw(m, sec, T_amb, f_cold),
            "gas_sendout_kg_per_s":   self._model.gas_sendout_kg_per_s(m),
            "gas_sendout_m3_per_day": self._model.gas_sendout_m3_per_day(m),
            "net_sec_kwh_per_ton":    self._model.net_sec_kwh_per_ton(m, sec, T_amb, f_cold),
        }

    def get_info(self) -> dict:
        return {
            "name": "LNG Regasification Terminal",
            "ec_id": "EC190",
            "fidelity": "F1a",
            "description": "SEC-based LNG sendout model with optional cold energy recovery",
            "inputs": {
                "sendout_rate_ton_per_h": {"unit": "ton/h",    "range": [10.0, 5000.0]},
                "sec_kwh_per_ton":        {"unit": "kWh/ton",  "range": [20.0, 120.0], "default": 50.0},
                "T_ambient_K":            {"unit": "K",        "range": [260.0, 310.0], "default": 288.15},
                "f_cold":                 {"unit": "-",        "range": [0.0, 0.6],    "default": 0.0},
            },
            "outputs": {
                "power_demand_kw":        {"unit": "kW"},
                "cold_recovery_kw":       {"unit": "kW"},
                "net_power_kw":           {"unit": "kW"},
                "gas_sendout_kg_per_s":   {"unit": "kg/s"},
                "gas_sendout_m3_per_day": {"unit": "m³/day"},
                "net_sec_kwh_per_ton":    {"unit": "kWh/ton"},
            },
            "source": "Mokhatab et al. (2014); Shah et al. (2013)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for sec in [25, 50, 80, 100]:
        r = model.predict({"sendout_rate_ton_per_h": 500.0, "sec_kwh_per_ton": float(sec)})
        print(f"SEC={sec} kWh/ton: P={float(r['power_demand_kw']):.0f} kW, "
              f"gas={float(r['gas_sendout_kg_per_s']):.2f} kg/s, "
              f"gas={float(r['gas_sendout_m3_per_day'])/1e6:.3f} Mm³/day")
