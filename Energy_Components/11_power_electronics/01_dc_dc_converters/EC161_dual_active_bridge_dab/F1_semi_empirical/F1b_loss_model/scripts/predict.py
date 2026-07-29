"""EC161 -- Dual Active Bridge (DAB) -- F1b Detailed Loss Model -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import DABF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = DABF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            v_in          : float [V]  primary DC bus voltage
            v_out_target  : float [V]  secondary DC bus voltage
            p_load        : float or array [W]  output power
        returns:
            phi_rad, efficiency, p_loss_w, T_j_degC,
            p_mosfet_cond_w, p_switching_w, p_transformer_w
        """
        v_in = float(inputs["v_in"])
        v_out_t = float(inputs["v_out_target"])
        p_load = np.asarray(inputs["p_load"], dtype=float)

        phi = self._model.phase_shift(v_in, v_out_t, p_load)
        breakdown = self._model.loss_breakdown(v_in, v_out_t, p_load)
        eta = self._model.efficiency(v_in, v_out_t, p_load)
        p_loss = self._model.total_losses(v_in, v_out_t, p_load)
        T_j = self._model.junction_temperature(v_in, v_out_t, p_load)

        return {
            "phi_rad": phi,
            "efficiency": eta,
            "p_loss_w": p_loss,
            "T_j_degC": T_j,
            "p_mosfet_cond_w": breakdown["p_mosfet_cond_w"],
            "p_switching_w": breakdown["p_switching_w"],
            "p_transformer_w": breakdown["p_transformer_w"],
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Dual Active Bridge (DAB) DC-DC Converter",
            "ec_id": "EC161",
            "fidelity": "F1b",
            "description": (
                "SPS-modulated DAB loss model: 8 MOSFET conduction (I^2*Rds_on(T)), "
                "8 MOSFET switching (0.5*V*I*(t_on+t_off)*f_sw), "
                "transformer copper (I_rms^2*R_xfmr), thermal balance"
            ),
            "inputs": {
                "v_in": {"unit": "V", "range": [200.0, 800.0]},
                "v_out_target": {"unit": "V", "range": [100.0, 400.0]},
                "p_load": {"unit": "W", "range": [0.0, 15000.0]},
            },
            "outputs": {
                "phi_rad": {"unit": "rad"},
                "efficiency": {"unit": "dimensionless"},
                "p_loss_w": {"unit": "W"},
                "T_j_degC": {"unit": "degC"},
                "p_mosfet_cond_w": {"unit": "W"},
                "p_switching_w": {"unit": "W"},
                "p_transformer_w": {"unit": "W"},
            },
            "params": {
                "n_turns": f"{u['n_turns']['value']}:1",
                "L_s": f"{u['L_s']['value']*1e6:.1f} uH",
                "f_sw": f"{u['f_sw']['value']/1e3:.0f} kHz",
                "R_ds_on": f"{u['R_ds_on']['value']*1000:.0f} mohm at {u['T_ref']['value']} degC",
                "R_theta": f"{u['R_theta']['value']} degC/W",
            },
            "source": "De Doncker et al. (1991), IEEE Trans. Ind. Appl.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"v_in": 400.0, "v_out_target": 200.0, "p_load": 5000.0})
    print(f"phi={float(r['phi_rad']):.4f} rad  eta={float(r['efficiency'])*100:.2f}%  "
          f"T_j={float(r['T_j_degC']):.1f}C  P_loss={float(r['p_loss_w']):.1f}W")
