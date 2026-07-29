"""EC191 — Gas Compressor Station — F1b Part-Load + Inlet Temperature — Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import NGCompressorF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = NGCompressorF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict compressor performance with part-load efficiency degradation.

        Parameters
        ----------
        inputs : dict
            m_dot_kg_s  : float (kg/s, design-point mass flow)
            P_in_bar    : float (bar)
            P_out_bar   : float (bar)
            T_in_K      : float (K, default 288.15)
            PLR         : float (0.3-1.0)
        """
        m = inputs.get("m_dot_kg_s", 100.0)
        P_in = inputs.get("P_in_bar", 40.0)
        P_out = inputs.get("P_out_bar", 70.0)
        T_in = inputs.get("T_in_K", None)
        plr = np.asarray(inputs.get("PLR", 1.0), dtype=float)
        return self._model.compute(m, P_in, P_out, T_in, plr)

    def get_info(self) -> dict:
        return {
            "name": "Gas Compressor Station",
            "ec_id": "EC191",
            "fidelity": "F1b",
            "description": (
                "Multistage polytropic compressor with PLR-dependent efficiency "
                "degradation and inlet temperature correction (API 617 / ISO 13631)."
            ),
            "inputs": {
                "m_dot_kg_s": {"unit": "kg/s", "range": [1, 1000]},
                "P_in_bar": {"unit": "bar", "range": [10, 100]},
                "P_out_bar": {"unit": "bar", "range": [20, 200]},
                "T_in_K": {"unit": "K", "range": [250, 330], "default": 288.15},
                "PLR": {"unit": "dimensionless", "range": [0.3, 1.0], "default": 1.0},
            },
            "outputs": {
                "specific_work_kJ_per_kg": {"unit": "kJ/kg"},
                "sec_kwh_per_kg": {"unit": "kWh/kg"},
                "shaft_power_kw": {"unit": "kW"},
                "discharge_temperature_K": {"unit": "K"},
                "polytropic_efficiency": {"unit": "dimensionless"},
                "overall_efficiency": {"unit": "dimensionless"},
            },
            "source": "Menon (2005); Campbell (2014); API 617",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"m_dot_kg_s": 100.0, "P_in_bar": 40.0, "P_out_bar": 70.0,
                        "T_in_K": 288.15, "PLR": 1.0})
    print("Design point (PLR=1.0):")
    for k, v in r.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")
    r2 = model.predict({"m_dot_kg_s": 100.0, "P_in_bar": 40.0, "P_out_bar": 70.0,
                         "T_in_K": 310.0, "PLR": 0.5})
    print("\nPart-load (PLR=0.5, hot day T_in=310 K):")
    for k, v in r2.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")
