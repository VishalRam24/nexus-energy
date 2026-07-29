"""EC110 -- Reciprocating Gas Engine -- F1b Part-Load + Altitude + Ambient -- Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import ReciprocatingGasEngineF1b


class ComponentModel:
    """Standardized interface for EC110 Reciprocating Gas Engine -- F1b model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ReciprocatingGasEngineF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            PLR         : float or array [0.5 - 1.0]
            T_ambient   : float or array [degC] (default 25.0)
            altitude_m  : float or array [m] (default 0.0)
        returns:
            efficiency_electrical : electrical efficiency [-]
            power_electrical_kw   : electrical output [kW_e]
            fuel_input_kw         : fuel input [kW_fuel, LHV]
            gas_mass_flow_kgs     : natural gas mass flow [kg/s]
            gas_volume_flow_m3h   : natural gas volume flow [m^3/h]
            sfc_g_kwh             : specific fuel consumption [g/kWh]
            heat_rate_kj_kwh      : heat rate [kJ/kWh]
            f_temperature         : temperature derating factor [-]
            f_altitude            : altitude derating factor [-]
        """
        PLR  = np.asarray(inputs["PLR"], dtype=float)
        T    = np.asarray(inputs.get("T_ambient", 25.0), dtype=float)
        alt  = np.asarray(inputs.get("altitude_m", 0.0), dtype=float)

        return {
            "efficiency_electrical": self._model.eta_electrical(PLR),
            "power_electrical_kw":   self._model.power_electrical_kw(PLR, T, alt),
            "fuel_input_kw":         self._model.fuel_input_kw(PLR, T, alt),
            "gas_mass_flow_kgs":     self._model.gas_mass_flow_kgs(PLR, T, alt),
            "gas_volume_flow_m3h":   self._model.gas_volume_flow_m3h(PLR, T, alt),
            "sfc_g_kwh":             self._model.sfc_g_kwh(PLR, T, alt),
            "heat_rate_kj_kwh":      self._model.heat_rate_kj_kwh(PLR),
            "f_temperature":         self._model.f_temperature(T),
            "f_altitude":            self._model.f_altitude(alt),
        }

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Reciprocating Gas Engine",
            "ec_id": "EC110",
            "fidelity": "F1b",
            "model": "Part-Load + Altitude + Ambient Temperature Derating",
            "description": (
                f"eta_el(PLR) = {m.eta_el_rated} * (b0+b1*PLR+b2*PLR^2); "
                f"f_temp = 1 - {m.temp_derate*100:.1f}%/degC above {m.temp_start}C; "
                f"f_alt = 1 - {m.alt_derate*100:.1f}%/100m; "
                f"P_el = P_rated*PLR*f_temp*f_alt"
            ),
            "inputs": {
                "PLR":        {"unit": "-", "range": [0.5, 1.0]},
                "T_ambient":  {"unit": "degC", "range": [-20, 50], "default": 25.0},
                "altitude_m": {"unit": "m", "range": [0, 3000], "default": 0.0},
            },
            "outputs": {
                "efficiency_electrical": {"unit": "-"},
                "power_electrical_kw":   {"unit": "kW_e"},
                "fuel_input_kw":         {"unit": "kW_fuel"},
                "gas_mass_flow_kgs":     {"unit": "kg/s"},
                "gas_volume_flow_m3h":   {"unit": "m^3/h"},
                "sfc_g_kwh":             {"unit": "g/kWh"},
                "heat_rate_kj_kwh":      {"unit": "kJ/kWh"},
                "f_temperature":         {"unit": "-"},
                "f_altitude":            {"unit": "-"},
            },
            "source": "US EPA CHP Catalog (2017); ISO 3046-1:2002; Jenbacher J320/J420; Caterpillar G3500",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC110 F1b -- Standard (PLR=1.0, 25C, sea level):")
    r = model.predict({"PLR": 1.0})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
    print("\nPart-load, hot, high altitude (PLR=0.7, 40C, 1500m):")
    r = model.predict({"PLR": 0.7, "T_ambient": 40.0, "altitude_m": 1500.0})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
