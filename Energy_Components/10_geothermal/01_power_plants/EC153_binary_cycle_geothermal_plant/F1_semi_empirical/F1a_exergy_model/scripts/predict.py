"""EC153 — Binary Cycle Geothermal Plant — F1a Exergy Model — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import BinaryGeothermalF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BinaryGeothermalF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict binary cycle geothermal plant performance.

        Parameters
        ----------
        inputs : dict
            T_geothermal  : float or array  (degC, 80–200)
            T_rejection   : float or array  (degC, 10–40)
            flow_rate_kgs : float or array  (kg/s, 10–100)

        Returns
        -------
        dict
            power_kw           : net electrical output (kW)
            efficiency         : overall plant efficiency (-)
            heat_input_kw      : thermal energy from brine (kW)
            T_reinjection_c    : brine reinjection temperature (degC)
        """
        T_geo  = np.asarray(inputs["T_geothermal"],  dtype=float)
        T_rej  = np.asarray(inputs["T_rejection"],   dtype=float)
        m_dot  = np.asarray(inputs["flow_rate_kgs"], dtype=float)

        power   = self._model.power_output(T_geo, T_rej, m_dot)
        eta     = self._model.plant_efficiency(T_geo, T_rej)
        Q_in    = self._model.heat_input(T_geo, T_rej, m_dot)
        T_reinj = self._model.reinjection_temperature(T_rej)

        return {
            "power_kw": power,
            "efficiency": eta,
            "heat_input_kw": Q_in,
            "T_reinjection_c": T_reinj,
        }

    def get_info(self) -> dict:
        return {
            "name": "Binary Cycle Geothermal Plant",
            "ec_id": "EC153",
            "fidelity": "F1a",
            "description": (
                "Exergy model: eta = eta_util * eta_Carnot, "
                "P = m_dot * cp * (T_geo - T_reinject) * eta_plant"
            ),
            "inputs": {
                "T_geothermal":  {"unit": "degC",  "range": [80.0, 200.0]},
                "T_rejection":   {"unit": "degC",  "range": [10.0, 40.0]},
                "flow_rate_kgs": {"unit": "kg/s",  "range": [10.0, 100.0]},
            },
            "outputs": {
                "power_kw":        {"unit": "kW"},
                "efficiency":      {"unit": "dimensionless"},
                "heat_input_kw":   {"unit": "kW"},
                "T_reinjection_c": {"unit": "degC"},
            },
            "source": "DiPippo (2015), Geothermal Power Plants, 4th ed.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_geothermal": 150.0, "T_rejection": 25.0, "flow_rate_kgs": 50.0})
    print(f"T_geo=150°C, T_rej=25°C, m_dot=50 kg/s:")
    print(f"  Power       = {float(r['power_kw']):.1f} kW")
    print(f"  Efficiency  = {float(r['efficiency']):.4f} ({float(r['efficiency'])*100:.2f}%)")
    print(f"  Heat Input  = {float(r['heat_input_kw']):.1f} kW")
    print(f"  T_reinject  = {float(r['T_reinjection_c']):.1f} °C")
    info = model.get_info()
    print(f"  EC ID: {info['ec_id']}, Fidelity: {info['fidelity']}")
