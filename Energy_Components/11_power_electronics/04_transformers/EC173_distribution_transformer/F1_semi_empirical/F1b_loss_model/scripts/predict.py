"""EC173 -- Distribution Transformer -- F1b IEC Loss Model -- Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import DistributionTransformerF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = DistributionTransformerF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            load_fraction : float or array [pu]  load fraction (0=no load, 1=rated)
            voltage_pu    : float or array [pu]  applied voltage (default 1.0)
            winding_temp  : float or array [°C]  winding temperature (default 75.0)
            power_factor  : float or array        (default 1.0)
            ambient_temp  : float or array [°C]  ambient temperature (default 20.0)
        returns:
            efficiency, p_loss_w, p_core_w, p_copper_w, p_stray_w, t_hot_spot_degc
        """
        plr = np.asarray(inputs["load_fraction"], dtype=float)
        v_pu = np.asarray(inputs.get("voltage_pu", 1.0), dtype=float)
        T_w = np.asarray(inputs.get("winding_temp", 75.0), dtype=float)
        pf = np.asarray(inputs.get("power_factor", 1.0), dtype=float)
        T_amb = np.asarray(inputs.get("ambient_temp", 20.0), dtype=float)

        breakdown = self._model.loss_breakdown(plr, v_pu, T_w)
        eta = self._model.efficiency(plr, v_pu, T_w, pf)
        t_hot = self._model.hot_spot_temperature(plr, T_amb)

        return {
            "efficiency": eta,
            "p_loss_w": breakdown["p_total_w"],
            "p_core_w": breakdown["p_core_w"],
            "p_copper_w": breakdown["p_copper_w"],
            "p_stray_w": breakdown["p_stray_w"],
            "t_hot_spot_degc": t_hot,
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Distribution Transformer",
            "ec_id": "EC173",
            "fidelity": "F1b",
            "description": (
                "IEC 60076 loss model: core loss (Steinmetz voltage-dependent), "
                "copper loss (temperature-corrected I^2R), stray losses, "
                "and IEC 60076-7 hot-spot thermal model. "
                f"Optimal load = {self._model.optimal_load_fraction()*100:.0f}% rated."
            ),
            "inputs": {
                "load_fraction": {"unit": "pu", "range": [0.0, 1.5]},
                "voltage_pu": {"unit": "pu", "range": [0.9, 1.1]},
                "winding_temp": {"unit": "degC", "range": [20.0, 200.0]},
                "power_factor": {"unit": "dimensionless", "range": [0.5, 1.0]},
                "ambient_temp": {"unit": "degC", "range": [-10.0, 50.0]},
            },
            "outputs": {
                "efficiency": {"unit": "dimensionless"},
                "p_loss_w": {"unit": "W"},
                "p_core_w": {"unit": "W"},
                "p_copper_w": {"unit": "W"},
                "p_stray_w": {"unit": "W"},
                "t_hot_spot_degc": {"unit": "degC"},
            },
            "params": {
                "S_rated": f"{u['S_rated_kVA']['value']:.0f} kVA",
                "P_no_load": f"{u['P_no_load_W']['value']:.0f} W",
                "P_load_loss": f"{u['P_load_loss_W']['value']:.0f} W",
                "u_k": f"{u['u_k_pu']['value']*100:.0f}%",
            },
            "source": "IEC 60076-1:2011 / IEC 60076-7:2018",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"load_fraction": 1.0})
    print(f"eta={float(r['efficiency'])*100:.3f}%  P_loss={float(r['p_loss_w']):.1f}W  "
          f"T_hot={float(r['t_hot_spot_degc']):.1f}°C")
    r_opt = model.predict({"load_fraction": model._model.optimal_load_fraction()})
    print(f"Optimal PLR: {model._model.optimal_load_fraction():.3f}  "
          f"eta_max={float(r_opt['efficiency'])*100:.3f}%")
