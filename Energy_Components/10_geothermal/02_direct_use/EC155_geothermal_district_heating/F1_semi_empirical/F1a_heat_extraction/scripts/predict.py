"""EC155 — Geothermal District Heating — F1a Heat Extraction Model — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import GeothermalDistrictHeatingF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = GeothermalDistrictHeatingF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict geothermal district heating system performance.

        Parameters
        ----------
        inputs : dict
            T_source      : float or array  (degC, 50–150) — geothermal supply temperature
            T_return      : float or array  (degC, 20–60)  — return temperature to reinjection
            flow_rate_kgs : float or array  (kg/s, 5–500)  — geothermal fluid flow rate

        Returns
        -------
        dict
            heat_extracted_kw    : thermal power from geothermal fluid (kW)
            heat_transferred_kw  : heat across HX to network (kW)
            heat_delivered_kw    : heat to end users after distribution losses (kW)
            heat_coefficient     : Q_delivered / Q_extracted (dimensionless, ~0.85-0.95)
            pump_power_kw        : circulation pump consumption (kW)
            system_cop           : Q_delivered / W_pump (dimensionless)
        """
        T_src  = np.asarray(inputs["T_source"],       dtype=float)
        T_ret  = np.asarray(inputs["T_return"],        dtype=float)
        m_dot  = np.asarray(inputs["flow_rate_kgs"],   dtype=float)

        Q_ext   = self._model.heat_extracted(T_src, T_ret, m_dot)
        Q_trans = self._model.heat_transferred(T_src, T_ret, m_dot)
        Q_del   = self._model.heat_delivered(T_src, T_ret, m_dot)
        coeff   = self._model.heat_coefficient(T_src, T_ret, m_dot)
        W_pump  = self._model.pump_power(T_src, T_ret, m_dot)
        scop    = self._model.system_cop(T_src, T_ret, m_dot)

        return {
            "heat_extracted_kw":   Q_ext,
            "heat_transferred_kw": Q_trans,
            "heat_delivered_kw":   Q_del,
            "heat_coefficient":    coeff,
            "pump_power_kw":       W_pump,
            "system_cop":          scop,
        }

    def get_info(self) -> dict:
        return {
            "name": "Geothermal District Heating",
            "ec_id": "EC155",
            "fidelity": "F1a",
            "description": (
                "Heat extraction model: Q_del = m_dot*cp*(T_src-T_ret)*eta_HX*(1-f_dist); "
                "direct use, no power generation"
            ),
            "inputs": {
                "T_source":      {"unit": "degC",  "range": [50.0,  150.0]},
                "T_return":      {"unit": "degC",  "range": [20.0,  60.0]},
                "flow_rate_kgs": {"unit": "kg/s",  "range": [5.0,   500.0]},
            },
            "outputs": {
                "heat_extracted_kw":   {"unit": "kW"},
                "heat_transferred_kw": {"unit": "kW"},
                "heat_delivered_kw":   {"unit": "kW"},
                "heat_coefficient":    {"unit": "dimensionless"},
                "pump_power_kw":       {"unit": "kW"},
                "system_cop":          {"unit": "dimensionless"},
            },
            "source": "Lund & Toth (2021), Geothermics; Rybach (2003), Geothermics",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_source": 80.0, "T_return": 40.0, "flow_rate_kgs": 50.0})
    print(f"T_source=80°C, T_return=40°C, m_dot=50 kg/s:")
    print(f"  Q_extracted  = {float(r['heat_extracted_kw']):.1f} kW")
    print(f"  Q_delivered  = {float(r['heat_delivered_kw']):.1f} kW")
    print(f"  Coefficient  = {float(r['heat_coefficient']):.3f}")
    print(f"  Pump power   = {float(r['pump_power_kw']):.2f} kW")
    print(f"  System COP   = {float(r['system_cop']):.1f}")
    info = model.get_info()
    print(f"  EC ID: {info['ec_id']}, Fidelity: {info['fidelity']}")
