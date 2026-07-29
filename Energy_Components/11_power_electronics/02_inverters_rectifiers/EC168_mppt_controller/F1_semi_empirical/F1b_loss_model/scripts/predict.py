"""EC168 -- MPPT Controller -- F1b Loss Model -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import MPPTF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MPPTF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            irradiance       : float or array [W/m2]
            p_mpp_available  : float or array [W]
            dG_dt            : float or array [W/m2/s] (default 0)
        returns:
            eta_static, eta_dynamic, eta_converter, eta_total,
            p_out_w, p_loss_w,
            p_oscillation_loss_w, p_dynamic_loss_w, p_converter_loss_w
        """
        G = np.asarray(inputs["irradiance"], dtype=float)
        P_mpp = np.asarray(inputs["p_mpp_available"], dtype=float)
        dG_dt = np.asarray(inputs.get("dG_dt", 0.0), dtype=float)

        eta_s = self._model.static_tracking_efficiency(G, P_mpp)
        eta_d = self._model.dynamic_tracking_efficiency(G, dG_dt)
        eta_c = self._model.eta_converter
        eta_total = self._model.total_efficiency(G, P_mpp, dG_dt)
        p_out = self._model.output_power(G, P_mpp, dG_dt)
        p_loss = self._model.total_losses(G, P_mpp, dG_dt)
        bd = self._model.loss_breakdown(G, P_mpp, dG_dt)

        return {
            "eta_static": eta_s,
            "eta_dynamic": eta_d,
            "eta_converter": np.full_like(G, eta_c) if np.ndim(G) > 0 else eta_c,
            "eta_total": eta_total,
            "p_out_w": p_out,
            "p_loss_w": p_loss,
            "p_oscillation_loss_w": bd["p_oscillation_loss_w"],
            "p_dynamic_loss_w": bd["p_dynamic_loss_w"],
            "p_converter_loss_w": bd["p_converter_loss_w"],
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "MPPT Maximum Power Point Tracker",
            "ec_id": "EC168",
            "fidelity": "F1b",
            "description": (
                "P&O algorithm loss model: "
                "oscillation loss (V_step perturbation at MPP), "
                "dynamic tracking loss (irradiance transients), "
                "DC-DC converter conduction+switching losses"
            ),
            "inputs": {
                "irradiance": {"unit": "W/m2", "range": [0.0, 1200.0]},
                "p_mpp_available": {"unit": "W", "range": [0.0, 12000.0]},
                "dG_dt": {"unit": "W/m2/s", "range": [-500.0, 500.0]},
            },
            "outputs": {
                "eta_static": {"unit": "dimensionless"},
                "eta_dynamic": {"unit": "dimensionless"},
                "eta_converter": {"unit": "dimensionless"},
                "eta_total": {"unit": "dimensionless"},
                "p_out_w": {"unit": "W"},
                "p_loss_w": {"unit": "W"},
                "p_oscillation_loss_w": {"unit": "W"},
                "p_dynamic_loss_w": {"unit": "W"},
                "p_converter_loss_w": {"unit": "W"},
            },
            "params": {
                "V_step": f"{u['V_step']['value']} V",
                "T_mppt": f"{u['T_mppt']['value']*1000:.0f} ms",
                "P_max": f"{u['P_max']['value']/1e3:.0f} kW",
                "eta_static": f"{u['eta_static']['value']*100:.1f}%",
                "eta_converter": f"{u['eta_converter']['value']*100:.1f}%",
            },
            "source": "Hohm & Ropp (2003); Femia et al. (2005)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    # Steady state at STC
    r = model.predict({"irradiance": 1000.0, "p_mpp_available": 8000.0, "dG_dt": 0.0})
    print(f"Steady-state (G=1000, P_mpp=8kW):")
    print(f"  eta_static={float(r['eta_static'])*100:.2f}%  "
          f"eta_dynamic={float(r['eta_dynamic'])*100:.2f}%  "
          f"eta_total={float(r['eta_total'])*100:.2f}%")
    print(f"  P_out={float(r['p_out_w']):.1f}W  P_loss={float(r['p_loss_w']):.1f}W")

    # During cloud transient
    r2 = model.predict({"irradiance": 500.0, "p_mpp_available": 4000.0, "dG_dt": -200.0})
    print(f"\nDuring cloud (G=500, dG/dt=-200):")
    print(f"  eta_static={float(r2['eta_static'])*100:.2f}%  "
          f"eta_dynamic={float(r2['eta_dynamic'])*100:.2f}%  "
          f"eta_total={float(r2['eta_total'])*100:.2f}%")
