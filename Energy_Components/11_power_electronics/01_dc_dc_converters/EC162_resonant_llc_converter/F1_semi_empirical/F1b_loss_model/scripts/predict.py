"""EC162 -- Resonant LLC Converter -- F1b Detailed Loss Model -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import LLCConverterF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = LLCConverterF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            v_in          : float [V]
            v_out_target  : float [V]  (used for efficiency calculation)
            i_load        : float or array [A]
        returns:
            efficiency, p_loss_w, T_j_degC,
            p_mosfet_cond_w, p_switching_w, p_diode_cond_w,
            p_resonant_inductor_w, p_transformer_pri_w, p_transformer_sec_w
        """
        v_in = float(inputs["v_in"])
        v_out_t = np.asarray(inputs["v_out_target"], dtype=float)
        i_load = np.asarray(inputs["i_load"], dtype=float)

        breakdown = self._model.loss_breakdown(v_in, i_load)
        eta = self._model.efficiency(v_in, v_out_t, i_load)
        p_loss = self._model.total_losses(v_in, i_load)
        T_j = self._model.junction_temperature(v_in, i_load)

        return {
            "efficiency": eta,
            "p_loss_w": p_loss,
            "T_j_degC": T_j,
            "p_mosfet_cond_w": breakdown["p_mosfet_cond_w"],
            "p_switching_w": breakdown["p_switching_w"],
            "p_diode_cond_w": breakdown["p_diode_cond_w"],
            "p_resonant_inductor_w": breakdown["p_resonant_inductor_w"],
            "p_transformer_pri_w": breakdown["p_transformer_pri_w"],
            "p_transformer_sec_w": breakdown["p_transformer_sec_w"],
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Resonant LLC Converter",
            "ec_id": "EC162",
            "fidelity": "F1b",
            "description": (
                "LLC resonant loss model at/near resonance: "
                "MOSFET conduction (ZVS operation), "
                "residual turn-off switching, "
                "diode conduction, resonant inductor + transformer copper losses, "
                "thermal balance T_j=T_a+P*R_theta"
            ),
            "inputs": {
                "v_in": {"unit": "V", "range": [200.0, 600.0]},
                "v_out_target": {"unit": "V", "range": [5.0, 24.0]},
                "i_load": {"unit": "A", "range": [0.0, 200.0]},
            },
            "outputs": {
                "efficiency": {"unit": "dimensionless"},
                "p_loss_w": {"unit": "W"},
                "T_j_degC": {"unit": "degC"},
                "p_mosfet_cond_w": {"unit": "W"},
                "p_switching_w": {"unit": "W"},
                "p_diode_cond_w": {"unit": "W"},
                "p_resonant_inductor_w": {"unit": "W"},
                "p_transformer_pri_w": {"unit": "W"},
                "p_transformer_sec_w": {"unit": "W"},
            },
            "params": {
                "n_turns": f"{u['n_turns']['value']}:1",
                "f_r": f"{u['f_r']['value']/1e3:.0f} kHz",
                "f_sw": f"{u['f_sw']['value']/1e3:.0f} kHz",
                "R_ds_on": f"{u['R_ds_on']['value']*1000:.0f} mohm",
                "R_theta": f"{u['R_theta']['value']} degC/W",
            },
            "source": "Yang et al. (2002), IEEE APEC. LLC Resonant Converter.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"v_in": 400.0, "v_out_target": 12.0, "i_load": 50.0})
    print(f"eta={float(r['efficiency'])*100:.2f}%  T_j={float(r['T_j_degC']):.1f}C  "
          f"P_loss={float(r['p_loss_w']):.2f}W")
