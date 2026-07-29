"""EC208 — CO2 Geological Sequestration — F1b Injection Pressure + Leakage — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import CO2SequestrationF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CO2SequestrationF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict CO2 geological sequestration with pressure build-up and leakage.

        Parameters
        ----------
        inputs : dict
            P_wellhead_bar      : float (bar, default 150)
            m_injected_tonnes   : float (tCO2 already injected, default 0)
            injection_years     : float (years of injection, default 0)
        """
        P_wh = inputs.get("P_wellhead_bar", 150.0)
        m_inj = inputs.get("m_injected_tonnes", 0.0)
        t_years = inputs.get("injection_years", 0.0)

        return self._model.compute(P_wh, m_inj, t_years)

    def get_info(self) -> dict:
        return {
            "name": "CO2 Geological Sequestration",
            "ec_id": "EC208",
            "fidelity": "F1b",
            "description": (
                "CO2 injection with reservoir pressure build-up (tank model), "
                "fracture pressure constraint, and caprock leakage model (0.1%/yr). "
                "Injection rate declines as reservoir fills."
            ),
            "inputs": {
                "P_wellhead_bar": {"unit": "bar", "range": [80, 300], "default": 150},
                "m_injected_tonnes": {"unit": "tCO2", "range": [0, 1e9], "default": 0},
                "injection_years": {"unit": "years", "range": [0, 50], "default": 0},
            },
            "outputs": {
                "reservoir_pressure_bar": {"unit": "bar"},
                "injection_rate_kg_s": {"unit": "kg/s"},
                "injection_rate_tco2_per_day": {"unit": "tCO2/day"},
                "cumulative_leakage_tco2": {"unit": "tCO2"},
                "net_retention_fraction": {"unit": "dimensionless"},
                "pressure_buildup_pct": {"unit": "%"},
                "max_wellhead_pressure_bar": {"unit": "bar"},
            },
            "source": "IPCC (2005) CCS Ch.5; Nordbotten et al. (2005); van der Meer (1993)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"P_wellhead_bar": 150.0, "m_injected_tonnes": 0, "injection_years": 0})
    print("Initial injection (empty reservoir):")
    for k, v in r.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")
