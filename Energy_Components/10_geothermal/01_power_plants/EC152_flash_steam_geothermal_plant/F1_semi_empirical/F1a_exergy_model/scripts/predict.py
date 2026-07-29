"""EC152 — Flash Steam Geothermal Plant — F1a Exergy Model — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import FlashSteamGeothermalF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = FlashSteamGeothermalF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict flash steam geothermal plant performance.

        Parameters
        ----------
        inputs : dict
            T_geothermal  : float or array  (degC, 200–320) — brine wellhead temperature
            T_rejection   : float or array  (degC, 10–60)   — cooling rejection temperature
            flow_rate_kgs : float or array  (kg/s, 10–500)  — total brine mass flow rate
            T_flash       : float or array  (degC, optional) — flash separator temperature;
                            if omitted, uses optimal (geometric mean)

        Returns
        -------
        dict
            power_kw           : net electrical output (kW)
            efficiency         : overall plant efficiency (-)
            heat_input_kw      : thermal energy from brine flash (kW)
            T_flash_c          : flash separator temperature (degC)
            steam_quality      : steam dryness fraction after flash (-)
            T_condenser_c      : condenser saturation temperature (degC)
        """
        T_geo  = np.asarray(inputs["T_geothermal"],  dtype=float)
        T_rej  = np.asarray(inputs["T_rejection"],   dtype=float)
        m_dot  = np.asarray(inputs["flow_rate_kgs"], dtype=float)
        T_flash_in = inputs.get("T_flash", None)
        T_flash = None if T_flash_in is None else np.asarray(T_flash_in, dtype=float)

        T_fl    = self._model.optimal_flash_temperature(T_geo, T_rej) if T_flash is None else T_flash
        power   = self._model.power_output(T_geo, T_rej, m_dot, T_flash)
        eta     = self._model.plant_efficiency(T_geo, T_rej)
        Q_in    = self._model.heat_input(T_geo, T_rej, m_dot, T_flash)
        x_steam = self._model.steam_quality(T_geo, T_fl)
        T_cond  = self._model.condenser_temperature(T_rej)

        return {
            "power_kw": power,
            "efficiency": eta,
            "heat_input_kw": Q_in,
            "T_flash_c": T_fl,
            "steam_quality": x_steam,
            "T_condenser_c": T_cond,
        }

    def get_info(self) -> dict:
        return {
            "name": "Flash Steam Geothermal Plant",
            "ec_id": "EC152",
            "fidelity": "F1a",
            "description": (
                "Exergy model with optimal flash temperature: "
                "T_flash=sqrt(T_geo*T_cond), eta=eta_util*eta_Carnot"
            ),
            "inputs": {
                "T_geothermal":  {"unit": "degC",  "range": [200.0, 320.0]},
                "T_rejection":   {"unit": "degC",  "range": [10.0,  60.0]},
                "flow_rate_kgs": {"unit": "kg/s",  "range": [10.0,  500.0]},
                "T_flash":       {"unit": "degC",  "range": [120.0, 200.0], "default": "optimal"},
            },
            "outputs": {
                "power_kw":      {"unit": "kW"},
                "efficiency":    {"unit": "dimensionless"},
                "heat_input_kw": {"unit": "kW"},
                "T_flash_c":     {"unit": "degC"},
                "steam_quality": {"unit": "dimensionless"},
                "T_condenser_c": {"unit": "degC"},
            },
            "source": "DiPippo (2015), Geothermal Power Plants, 4th ed., Chapters 5-6",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_geothermal": 240.0, "T_rejection": 40.0, "flow_rate_kgs": 100.0})
    print(f"T_geo=240°C, T_rej=40°C, m_dot=100 kg/s:")
    print(f"  Power        = {float(r['power_kw']):.1f} kW")
    print(f"  Efficiency   = {float(r['efficiency']):.4f} ({float(r['efficiency'])*100:.2f}%)")
    print(f"  Heat Input   = {float(r['heat_input_kw']):.1f} kW")
    print(f"  T_flash      = {float(r['T_flash_c']):.1f} °C")
    print(f"  Steam Quality= {float(r['steam_quality']):.3f}")
    info = model.get_info()
    print(f"  EC ID: {info['ec_id']}, Fidelity: {info['fidelity']}")
