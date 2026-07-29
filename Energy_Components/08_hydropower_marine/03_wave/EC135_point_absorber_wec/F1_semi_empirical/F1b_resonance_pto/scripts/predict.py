"""EC135 — Point Absorber WEC — F1b Resonance / PTO — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import PointAbsorberF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = PointAbsorberF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            H_s     : significant wave height [m]
            T_e     : energy period [s]
            T_C     : seawater temperature [degC] (optional)
            S_psu   : salinity [psu] (optional)
        returns:
            power_kw             : electrical output [kW] (limited to P_rated)
            cwr                  : capture width ratio at given T_e [-]
            pto_efficiency       : PTO eta at current power level [-]
            overall_efficiency   : wave-to-wire efficiency [-]
            seawater_density_kgm3: rho [kg/m3]
        """
        H_s   = np.asarray(inputs["H_s"], dtype=float)
        T_e   = np.asarray(inputs["T_e"], dtype=float)
        T_C   = inputs.get("T_C", None)
        S_psu = inputs.get("S_psu", None)
        m     = self._model
        P_kw  = m.power_kw(H_s, T_e, T_C, S_psu)
        cwr   = m.capture_width_ratio(T_e)
        J     = m.wave_power_per_metre(H_s, T_e, T_C, S_psu)
        P_m   = J * m.diameter * cwr / 1e3
        return {
            "power_kw":              P_kw,
            "cwr":                   cwr,
            "pto_efficiency":        m.pto_efficiency(P_m),
            "overall_efficiency":    m.overall_efficiency(H_s, T_e, T_C, S_psu),
            "seawater_density_kgm3": m.seawater_density(T_C, S_psu),
        }

    def get_info(self) -> dict:
        m = self._model
        return {
            "name":     "Point Absorber WEC",
            "ec_id":    "EC135",
            "fidelity": "F1b",
            "model":    "Resonance + PTO Part-Load Efficiency + Power Limiting",
            "description": (
                f"Heaving buoy: resonance CWR (peak={m.cwr_peak:.2f} at T_n={m.T_n:.0f}s, "
                f"sigma={m.sigma:.0f}s); PTO eta({m.eta_pto_min:.2f}-{m.eta_pto_rated:.2f}); "
                f"P_rated={m.P_rated_kw:.0f}kW; cutout at H_s={m.H_s_cutout:.1f}m."
            ),
            "inputs": {
                "H_s":   {"unit": "m",    "range": [0.0, 8.0]},
                "T_e":   {"unit": "s",    "range": [4.0, 20.0]},
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
            "source": "Falnes (2002); Babarit et al. (2012) Renew. Energy 41; CorPower (2020)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"H_s": 2.0, "T_e": 10.0})
    print(f"H_s=2m T_e=10s: P={float(r['power_kw']):.2f} kW, "
          f"CWR={float(r['cwr']):.3f}, eta_pto={float(r['pto_efficiency']):.3f}")
    r2 = model.predict({"H_s": 7.0, "T_e": 10.0})
    print(f"H_s=7m (storm): P={float(r2['power_kw']):.2f} kW (cutout expected)")
