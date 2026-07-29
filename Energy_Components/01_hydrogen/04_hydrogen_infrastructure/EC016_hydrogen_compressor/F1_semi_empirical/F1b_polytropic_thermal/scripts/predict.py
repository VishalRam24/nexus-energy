"""
EC016 -- H2 Compressor -- F1b Polytropic Thermal
ComponentModel wrapper with standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import H2CompressorThermalModel

_PARAMS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "parameters.json"
)


def _load_params() -> dict:
    with open(_PARAMS_PATH, "r") as fh:
        return json.load(fh)


class ComponentModel:
    """
    Standardised wrapper for the H2 compressor thermal model.

    Usage
    -----
    >>> model = ComponentModel()
    >>> out = model.predict({"mass_flow": 0.014, "P_inlet": 20.0, "P_outlet": 900.0,
    ...                      "T_inlet": 298.15})
    """

    component_id   = "EC016"
    component_name = "Hydrogen Compressor"
    fidelity       = "F1b -- Polytropic Thermal (T_inlet variation, intercooler effectiveness, stage T)"
    version        = "1.0.0"

    def __init__(self, params: dict = None):
        defaults = _load_params()
        self._raw = defaults
        self._physics = H2CompressorThermalModel(defaults)

    def predict(self, inputs: dict) -> dict:
        """
        Predict compressor performance.

        Parameters
        ----------
        inputs : dict
            mass_flow  : float -- kg/s (0 – 0.05)
            P_inlet    : float -- bar  (1 – 100)
            P_outlet   : float -- bar  (10 – 1000)
            T_inlet    : float -- K    (263 – 333), optional
            T_coolant  : float -- K    (263 – 333), optional
            intercooler_effectiveness : float (0 – 1), optional

        Returns
        -------
        dict with shaft_power_kW, SEC_kWh_kg, efficiency, heat_rejected_kW,
             T_discharge_final_K, stage_T_in, stage_T_discharge, stage_T_after_ic
        """
        m_dot  = float(inputs["mass_flow"])
        P_in   = float(inputs["P_inlet"])
        P_out  = float(inputs["P_outlet"])
        T_in   = float(inputs.get("T_inlet",   self._physics.T_inlet_default))
        T_cool = float(inputs.get("T_coolant", self._physics.T_cool))
        eps    = float(inputs.get("intercooler_effectiveness", self._physics.eps_ic))

        if not (0.0 <= m_dot <= 0.05):
            raise ValueError(f"mass_flow must be in [0, 0.05] kg/s, got {m_dot}")
        if not (1.0 <= P_in <= 100.0):
            raise ValueError(f"P_inlet must be in [1, 100] bar, got {P_in}")
        if not (10.0 <= P_out <= 1000.0):
            raise ValueError(f"P_outlet must be in [10, 1000] bar, got {P_out}")
        if P_out <= P_in:
            raise ValueError(f"P_outlet ({P_out}) must be > P_inlet ({P_in})")
        if not (263.0 <= T_in <= 333.0):
            raise ValueError(f"T_inlet must be in [263, 333] K, got {T_in}")
        if not (0.0 <= eps <= 1.0):
            raise ValueError(f"intercooler_effectiveness must be in [0, 1], got {eps}")

        res = self._physics.evaluate(m_dot, P_in, P_out, T_in, T_cool, eps)
        prof = res["stage_profile"]

        return {
            "shaft_power_kW":        round(float(res["shaft_power_kW"]),       4),
            "SEC_kWh_kg":            round(float(res["SEC_kWh_kg"]),            6),
            "efficiency":            round(float(res["efficiency"]),            6),
            "heat_rejected_kW":      round(float(res["heat_rejected_kW"]),      4),
            "T_discharge_final_K":   round(float(res["T_discharge_final_K"]),   4),
            "stage_T_in_K":          [round(float(t), 3) for t in prof["T_in_stage"]],
            "stage_T_discharge_K":   [round(float(t), 3) for t in prof["T_discharge"]],
            "stage_T_after_ic_K":    [round(float(t), 3) for t in prof["T_after_ic"]],
        }

    def get_info(self) -> dict:
        p = self._physics
        return {
            "component_id":   self.component_id,
            "component_name": self.component_name,
            "fidelity":       self.fidelity,
            "version":        self.version,
            "inputs": {
                "mass_flow":                 {"unit": "kg/s", "range": [0.0, 0.05]},
                "P_inlet":                   {"unit": "bar",  "range": [1.0, 100.0]},
                "P_outlet":                  {"unit": "bar",  "range": [10.0, 1000.0]},
                "T_inlet":                   {"unit": "K",    "range": [263.0, 333.0], "optional": True},
                "T_coolant":                 {"unit": "K",    "range": [263.0, 333.0], "optional": True},
                "intercooler_effectiveness": {"unit": "-",    "range": [0.0, 1.0],    "optional": True},
            },
            "outputs": {
                "shaft_power_kW":       "kW",
                "SEC_kWh_kg":           "kWh/kg_H2",
                "efficiency":           "-",
                "heat_rejected_kW":     "kW",
                "T_discharge_final_K":  "K",
                "stage_T_in_K":         "K (list, N stages)",
                "stage_T_discharge_K":  "K (list, N stages)",
                "stage_T_after_ic_K":   "K (list, N stages)",
            },
            "source": "Sdanghi et al. (2019); Bossel (2006); Aungier (2000)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print(model.get_info())
    print(model.predict({"mass_flow": 0.014, "P_inlet": 20.0, "P_outlet": 900.0,
                         "T_inlet": 298.15}))
    print("Hot inlet:", model.predict({"mass_flow": 0.014, "P_inlet": 20.0, "P_outlet": 900.0,
                                       "T_inlet": 328.15}))
