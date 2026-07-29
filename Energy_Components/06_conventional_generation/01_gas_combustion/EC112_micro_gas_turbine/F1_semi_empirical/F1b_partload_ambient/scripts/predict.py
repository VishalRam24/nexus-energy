"""EC112 -- Micro Gas Turbine -- F1b Part-Load + Ambient + Altitude -- Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import MicroGasTurbineF1b


class ComponentModel:
    """Standardized interface for EC112 Micro Gas Turbine -- F1b model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MicroGasTurbineF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            PLR         : float or array [0.3 - 1.0]
            T_ambient   : float or array [K] (default 288.15 = 15 degC ISO)
            P_ambient   : float or array [kPa] (default 101.325)
            altitude_m  : float or array [m] (default 0.0)
        returns:
            efficiency_electrical : net electrical efficiency [-]
            power_electrical_kw   : electrical output [kW_e]
            fuel_input_kw         : fuel input [kW_fuel, LHV]
            gas_mass_flow_kgs     : natural gas mass flow [kg/s]
            gas_volume_flow_m3h   : natural gas volume flow [m^3/h]
            heat_rate_kj_kwh      : heat rate [kJ/kWh]
            f_power_ambient       : ISO power correction factor [-]
            f_eta_temperature     : efficiency temperature correction factor [-]
            f_altitude            : altitude power correction factor [-]
        """
        PLR   = np.asarray(inputs["PLR"], dtype=float)
        T_amb = np.asarray(inputs.get("T_ambient", 288.15), dtype=float)
        P_amb = np.asarray(inputs.get("P_ambient", 101.325), dtype=float)
        alt   = np.asarray(inputs.get("altitude_m", 0.0), dtype=float)

        return {
            "efficiency_electrical": self._model.eta_electrical(PLR, T_amb),
            "power_electrical_kw":   self._model.power_electrical_kw(PLR, T_amb, P_amb, alt),
            "fuel_input_kw":         self._model.fuel_input_kw(PLR, T_amb, P_amb, alt),
            "gas_mass_flow_kgs":     self._model.gas_mass_flow_kgs(PLR, T_amb, P_amb, alt),
            "gas_volume_flow_m3h":   self._model.gas_volume_flow_m3h(PLR, T_amb, P_amb, alt),
            "heat_rate_kj_kwh":      self._model.heat_rate_kj_kwh(PLR, T_amb),
            "f_power_ambient":       self._model.f_power_ambient(T_amb, P_amb),
            "f_eta_temperature":     self._model.f_eta_temperature(T_amb),
            "f_altitude":            self._model.f_altitude(alt),
        }

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Micro Gas Turbine",
            "ec_id": "EC112",
            "fidelity": "F1b",
            "model": "Part-Load + Ambient Temperature (strong, ~0.01/K) + Altitude",
            "description": (
                f"eta_el(PLR,T) = {m.eta_el_rated} * (b0+b1*PLR+b2*PLR^2) * (1 - {m.f_amb_coeff}/K*(T-{m.T_ref}K)); "
                f"P_el = P_rated*PLR*(P/P_ref)*sqrt(T_ref/T)*f_alt; "
                f"Strong T sensitivity: 1%/K degradation above ISO ref."
            ),
            "inputs": {
                "PLR":        {"unit": "-", "range": [0.3, 1.0]},
                "T_ambient":  {"unit": "K", "range": [248.15, 323.15], "default": 288.15},
                "P_ambient":  {"unit": "kPa", "range": [85.0, 107.0], "default": 101.325},
                "altitude_m": {"unit": "m", "range": [0.0, 3000.0], "default": 0.0},
            },
            "outputs": {
                "efficiency_electrical": {"unit": "-"},
                "power_electrical_kw":   {"unit": "kW_e"},
                "fuel_input_kw":         {"unit": "kW_fuel"},
                "gas_mass_flow_kgs":     {"unit": "kg/s"},
                "gas_volume_flow_m3h":   {"unit": "m^3/h"},
                "heat_rate_kj_kwh":      {"unit": "kJ/kWh"},
                "f_power_ambient":       {"unit": "-"},
                "f_eta_temperature":     {"unit": "-"},
                "f_altitude":            {"unit": "-"},
            },
            "source": "US EPA CHP Catalog (2017); Capstone C200 datasheet; ISO 2314:2009",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC112 F1b -- ISO conditions (PLR=1.0, 15C=288.15K, 101.325 kPa, sea level):")
    r = model.predict({"PLR": 1.0})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
    print("\nPart-load, hot, high altitude (PLR=0.7, 40C=313.15K, 1500m):")
    r = model.predict({"PLR": 0.7, "T_ambient": 313.15, "altitude_m": 1500.0})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
