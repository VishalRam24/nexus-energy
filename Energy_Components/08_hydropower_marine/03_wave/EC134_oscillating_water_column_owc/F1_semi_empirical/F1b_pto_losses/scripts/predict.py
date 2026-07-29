"""EC134 — OWC — F1b PTO Losses — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import OWCF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = OWCF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            H_s     : significant wave height [m]
            T_e     : energy period [s]
            T_C     : seawater temperature [degC] (optional)
            S_psu   : salinity [psu] (optional)
        returns:
            power_kw             : electrical output [kW]
            turbine_efficiency   : eta_turbine at given T_e [-]
            overall_efficiency   : wave-to-wire efficiency [-]
            seawater_density_kgm3: rho [kg/m3]
        """
        H_s   = np.asarray(inputs["H_s"], dtype=float)
        T_e   = np.asarray(inputs["T_e"], dtype=float)
        T_C   = inputs.get("T_C", None)
        S_psu = inputs.get("S_psu", None)
        return {
            "power_kw":              self._model.power_kw(H_s, T_e, T_C, S_psu),
            "turbine_efficiency":    self._model.turbine_efficiency(T_e),
            "overall_efficiency":    self._model.overall_efficiency(T_e),
            "seawater_density_kgm3": self._model.seawater_density(T_C, S_psu),
        }

    def get_info(self) -> dict:
        m = self._model
        return {
            "name":     "Oscillating Water Column (OWC)",
            "ec_id":    "EC134",
            "fidelity": "F1b",
            "model":    "PTO Efficiency + Density Correction",
            "description": (
                f"OWC wave-to-wire model: Wells turbine eta vs T_e "
                f"(peak={m.eta_turb_peak:.2f} at T_e={m.T_e_design:.0f}s, "
                f"min={m.eta_turb_min:.2f}); "
                f"seawater density correction; "
                f"directional spreading factor={m.dir_factor:.2f}. "
                f"Device width={m.width:.1f}m, CWR={m.cwr:.2f}."
            ),
            "inputs": {
                "H_s":   {"unit": "m",    "range": [0.0, 8.0]},
                "T_e":   {"unit": "s",    "range": [4.0, 22.0]},
                "T_C":   {"unit": "degC", "optional": True},
                "S_psu": {"unit": "psu",  "optional": True},
            },
            "outputs": {
                "power_kw":              {"unit": "kW"},
                "turbine_efficiency":    {"unit": "dimensionless"},
                "overall_efficiency":    {"unit": "dimensionless"},
                "seawater_density_kgm3": {"unit": "kg/m3"},
            },
            "source": "Falnes (2002); Folley (2016); EMEC TR-001 (2019)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"H_s": 3.0, "T_e": 10.0, "T_C": 15.0, "S_psu": 35.0})
    print(f"Design (H_s=3m, T_e=10s): P={float(r['power_kw']):.2f} kW, "
          f"eta_turb={float(r['turbine_efficiency']):.3f}")
    r2 = model.predict({"H_s": 3.0, "T_e": 6.0})
    print(f"Off-design (T_e=6s):       P={float(r2['power_kw']):.2f} kW, "
          f"eta_turb={float(r2['turbine_efficiency']):.3f}")
