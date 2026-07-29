"""EC137 — Oscillating Body Attenuator WEC — F1b PTO Losses — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import AttenuatorWECF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AttenuatorWECF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            H_s     : significant wave height [m]
            T_e     : energy period [s]
            T_C     : seawater temperature [degC] (optional)
            S_psu   : salinity [psu] (optional)
        returns:
            power_kw              : electrical output [kW]
            cwr                   : CWR at given sea state [-]
            pto_efficiency        : hydraulic PTO eta [-]
            overall_efficiency    : wave-to-wire efficiency [-]
            seawater_density_kgm3 : rho [kg/m3]
        """
        H_s   = np.asarray(inputs["H_s"], dtype=float)
        T_e   = np.asarray(inputs["T_e"], dtype=float)
        T_C   = inputs.get("T_C", None)
        S_psu = inputs.get("S_psu", None)
        m     = self._model
        return {
            "power_kw":              m.power_kw(H_s, T_e, T_C, S_psu),
            "cwr":                   m.capture_width_ratio(H_s, T_e),
            "pto_efficiency":        m.pto_efficiency(H_s),
            "overall_efficiency":    m.overall_efficiency(H_s, T_e),
            "seawater_density_kgm3": m.seawater_density(T_C, S_psu),
        }

    def get_info(self) -> dict:
        m = self._model
        return {
            "name":     "Oscillating Body / Attenuator WEC (Pelamis-type)",
            "ec_id":    "EC137",
            "fidelity": "F1b",
            "model":    "Sea-State-Dependent CWR + Hydraulic PTO Efficiency",
            "description": (
                f"Attenuator WEC: CWR(H_s,T_e) = CWR_design*(H_s/H_s_d)^{m.cwr_Hs_exp:.2f}"
                f"*Gaussian(T_e; sigma={m.sigma_T:.1f}s); "
                f"hydraulic PTO eta={m.eta_pto_design:.2f} at design, "
                f"coeff={m.pto_Hs_coeff:.3f}/m; "
                f"dir_factor={m.dir_factor:.2f}; "
                f"L={m.length:.0f}m, {m.n_joints} joints."
            ),
            "inputs": {
                "H_s":   {"unit": "m",    "range": [0.0, 8.0]},
                "T_e":   {"unit": "s",    "range": [5.0, 20.0]},
                "T_C":   {"unit": "degC", "optional": True},
                "S_psu": {"unit": "psu",  "optional": True},
            },
            "outputs": {
                "power_kw":              {"unit": "kW"},
                "cwr":                   {"unit": "dimensionless"},
                "pto_efficiency":        {"unit": "dimensionless"},
                "overall_efficiency":    {"unit": "dimensionless"},
                "seawater_density_kgm3": {"unit": "kg/m3"},
            },
            "source": "Henderson (2006) Appl. Ocean Res. 28; Yemm et al. (2012) Phil. Trans. R. Soc. A 370",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"H_s": 3.0, "T_e": 10.0})
    print(f"Design (H_s=3m, T_e=10s): P={float(r['power_kw']):.2f} kW, "
          f"CWR={float(r['cwr']):.3f}, eta_pto={float(r['pto_efficiency']):.3f}")
