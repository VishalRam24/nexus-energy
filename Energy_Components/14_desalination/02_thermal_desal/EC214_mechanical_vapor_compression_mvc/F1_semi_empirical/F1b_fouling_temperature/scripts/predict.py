"""EC214 — MVC Desalination — F1b SEC + Recovery + Temperature — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import MVCF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MVCF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict MVC performance.

        Parameters
        ----------
        inputs : dict
            recovery        : float (0-0.75)     default 0.50
            T_evap_degC     : float (degC)        default 70.0
            operating_hours : float (hours)       default 0.0
            feed_flow_m3_h  : float (m3/h)       default 100.0
        """
        r     = np.asarray(inputs.get("recovery", 0.50), dtype=float)
        T_ev  = inputs.get("T_evap_degC", 70.0)
        hours = inputs.get("operating_hours", 0.0)
        Q_f   = inputs.get("feed_flow_m3_h", 100.0)

        return self._model.compute(r, T_ev, hours, Q_f)

    def get_info(self) -> dict:
        return {
            "name": "Mechanical Vapor Compression (MVC)",
            "ec_id": "EC214",
            "fidelity": "F1b",
            "description": (
                "MVC model with SEC vs recovery (BPE-corrected), compressor thermodynamics, "
                "temperature-dependent compressor work, and evaporator fouling "
                "(U declines ~10%/yr)."
            ),
            "inputs": {
                "recovery":       {"unit": "dimensionless", "range": [0.3, 0.75]},
                "T_evap_degC":    {"unit": "degC",          "range": [50, 80]},
                "operating_hours": {"unit": "hours",        "range": [0, 87600]},
                "feed_flow_m3_h": {"unit": "m3/h",          "range": [1, 1000]},
            },
            "outputs": {
                "sec_kwh_m3":             {"unit": "kWh_e/m3"},
                "compressor_work_kwh_m3": {"unit": "kWh_e/m3"},
                "production_rate_m3_h":   {"unit": "m3/h"},
                "brine_salinity_gkg":     {"unit": "g/kg"},
                "bpe_degC":               {"unit": "degC"},
                "fouling_factor":         {"unit": "dimensionless [0-1]"},
            },
            "source": "Mistry et al. (2011); Darwish (1988)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"recovery": 0.50, "T_evap_degC": 70.0, "operating_hours": 0})
    print("Design point (recovery=0.5, T_evap=70C, fresh):")
    for k, v in r.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")
