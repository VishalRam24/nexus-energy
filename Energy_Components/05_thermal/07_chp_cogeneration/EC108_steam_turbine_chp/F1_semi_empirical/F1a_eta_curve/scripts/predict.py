"""EC108 -- Steam Turbine CHP -- F1a Efficiency-Curve -- Standardized Predict Interface"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import SteamTurbineCHPF1a


class ComponentModel:
    component_id = "EC108"

    def __init__(self, params_path=None):
        if params_path is None:
            params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SteamTurbineCHPF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            PLR : float  -- part-load ratio [0.3 – 1.0]
        returns:
            PLR_clamped, P_el_kw, Q_th_kw, P_fuel_kw,
            eta_el, eta_th, eta_total, HPR
        """
        PLR = float(inputs.get("PLR", 1.0))
        return self._model.predict(PLR)

    def get_info(self) -> dict:
        m = self._model
        return {
            "name":        "Steam Turbine CHP (Back-Pressure)",
            "ec_id":       "EC108",
            "fidelity":    "F1a",
            "model":       "Constant-Efficiency Curve",
            "description": (
                f"Back-pressure steam turbine CHP. "
                f"P_el_rated={m.P_el_rated:.0f} kW_e, "
                f"eta_el={m.eta_el:.2f}, eta_th={m.eta_th:.2f}, "
                f"eta_total={m.eta_el+m.eta_th:.2f}. "
                f"PLR range [{m.PLR_min:.1f}, {m.PLR_max:.1f}]."
            ),
            "inputs":  {"PLR": {"unit": "dimensionless", "range": [0.3, 1.0]}},
            "outputs": {
                "PLR_clamped": {"unit": "dimensionless"},
                "P_el_kw":     {"unit": "kW_e"},
                "Q_th_kw":     {"unit": "kW_th"},
                "P_fuel_kw":   {"unit": "kW"},
                "eta_el":      {"unit": "dimensionless"},
                "eta_th":      {"unit": "dimensionless"},
                "eta_total":   {"unit": "dimensionless"},
                "HPR":         {"unit": "dimensionless"},
            },
            "source": "US EPA CHP Catalog (2017); IEA (2008) Combined Heat and Power",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")
    for PLR in [0.3, 0.5, 0.75, 1.0]:
        r = model.predict({"PLR": PLR})
        print(f"  PLR={PLR:.2f}: P_el={r['P_el_kw']:.0f} kW, Q_th={r['Q_th_kw']:.0f} kW, eta_total={r['eta_total']:.2f}")
