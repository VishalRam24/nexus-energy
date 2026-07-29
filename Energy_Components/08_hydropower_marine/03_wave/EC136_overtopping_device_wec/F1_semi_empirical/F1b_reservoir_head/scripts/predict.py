"""EC136 — Overtopping Device WEC — F1b Reservoir Head — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import OvertoppingWECF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = OvertoppingWECF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            H_s     : significant wave height [m]
            T_e     : energy period [s]
            T_C     : seawater temperature [degC] (optional)
            S_psu   : salinity [psu] (optional)
        returns:
            power_kw              : electrical output via wave path [kW]
            reservoir_head_m      : effective reservoir head [m]
            turbine_efficiency    : eta_turbine at h(H_s) [-]
            overall_efficiency    : wave-to-wire efficiency [-]
            seawater_density_kgm3 : rho [kg/m3]
        """
        H_s   = np.asarray(inputs["H_s"], dtype=float)
        T_e   = np.asarray(inputs["T_e"], dtype=float)
        T_C   = inputs.get("T_C", None)
        S_psu = inputs.get("S_psu", None)
        m     = self._model
        h     = m.reservoir_head_m(H_s)
        return {
            "power_kw":              m.power_kw(H_s, T_e, T_C, S_psu),
            "reservoir_head_m":      h,
            "turbine_efficiency":    m.turbine_efficiency(h),
            "overall_efficiency":    m.overall_efficiency(H_s),
            "seawater_density_kgm3": m.seawater_density(T_C, S_psu),
        }

    def get_info(self) -> dict:
        m = self._model
        return {
            "name":     "Overtopping Device WEC",
            "ec_id":    "EC136",
            "fidelity": "F1b",
            "model":    "Variable Reservoir Head + Turbine Efficiency",
            "description": (
                f"Overtopping WEC: reservoir head h(H_s) = h_design*(H_s/H_s_ref)^{m.Hs_exp}; "
                f"Kaplan turbine eta_peak={m.eta_turb_peak:.2f}; "
                f"density correction; ramp width={m.width:.0f}m."
            ),
            "inputs": {
                "H_s":   {"unit": "m",    "range": [0.5, 6.0]},
                "T_e":   {"unit": "s",    "range": [5.0, 20.0]},
                "T_C":   {"unit": "degC", "optional": True},
                "S_psu": {"unit": "psu",  "optional": True},
            },
            "outputs": {
                "power_kw":              {"unit": "kW"},
                "reservoir_head_m":      {"unit": "m"},
                "turbine_efficiency":    {"unit": "dimensionless"},
                "overall_efficiency":    {"unit": "dimensionless"},
                "seawater_density_kgm3": {"unit": "kg/m3"},
            },
            "source": "Kofoed et al. (2006) Coastal Eng. 53; Wave Dragon (2010)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"H_s": 2.0, "T_e": 10.0})
    print(f"H_s=2m, T_e=10s: P={float(r['power_kw']):.2f} kW, "
          f"h={float(r['reservoir_head_m']):.2f}m, "
          f"eta_turb={float(r['turbine_efficiency']):.3f}")
