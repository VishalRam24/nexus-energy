"""EC160 -- Isolated DC-DC (Flyback/Forward) -- F1b Detailed Loss Model -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import IsolatedDCDCF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = IsolatedDCDCF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            v_in          : float or array [V]
            v_out_target  : float or array [V]
            i_load        : float or array [A]
        returns:
            duty_cycle, v_out, efficiency, p_loss_w, T_j_degC,
            p_mosfet_cond_w, p_diode_cond_w, p_switching_w,
            p_transformer_pri_w, p_transformer_sec_w
        """
        v_in = np.asarray(inputs["v_in"], dtype=float)
        v_out_t = np.asarray(inputs["v_out_target"], dtype=float)
        i_load = np.asarray(inputs["i_load"], dtype=float)

        D = self._model.duty_cycle(v_in, v_out_t)
        breakdown = self._model.loss_breakdown(v_in, v_out_t, i_load)
        eta = self._model.efficiency(v_in, v_out_t, i_load)
        p_loss = self._model.total_losses(v_in, v_out_t, i_load)
        T_j = self._model.junction_temperature(v_in, v_out_t, i_load)

        return {
            "duty_cycle": D,
            "v_out": v_out_t,
            "efficiency": eta,
            "p_loss_w": p_loss,
            "T_j_degC": T_j,
            "p_mosfet_cond_w": breakdown["p_mosfet_cond_w"],
            "p_diode_cond_w": breakdown["p_diode_cond_w"],
            "p_switching_w": breakdown["p_switching_w"],
            "p_transformer_pri_w": breakdown["p_transformer_pri_w"],
            "p_transformer_sec_w": breakdown["p_transformer_sec_w"],
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Isolated DC-DC Converter (Flyback/Forward)",
            "ec_id": "EC160",
            "fidelity": "F1b",
            "description": (
                "Detailed loss model for isolated converters: "
                "MOSFET conduction (I_pri_rms^2*R_ds_on(T)), "
                "diode conduction (I_out*V_f), "
                "switching (0.5*V_drain*I_pk*(t_on+t_off)*f_sw), "
                "transformer primary + secondary copper losses, "
                "thermal balance T_j=T_a+P*R_theta"
            ),
            "inputs": {
                "v_in": {"unit": "V", "range": [20.0, 100.0]},
                "v_out_target": {"unit": "V", "range": [1.0, 30.0]},
                "i_load": {"unit": "A", "range": [0.0, 15.0]},
            },
            "outputs": {
                "duty_cycle": {"unit": "dimensionless"},
                "v_out": {"unit": "V"},
                "efficiency": {"unit": "dimensionless"},
                "p_loss_w": {"unit": "W"},
                "T_j_degC": {"unit": "degC"},
                "p_mosfet_cond_w": {"unit": "W"},
                "p_diode_cond_w": {"unit": "W"},
                "p_switching_w": {"unit": "W"},
                "p_transformer_pri_w": {"unit": "W"},
                "p_transformer_sec_w": {"unit": "W"},
            },
            "params": {
                "n_turns": f"{u['n_turns']['value']}:1",
                "f_sw": f"{u['f_sw']['value']/1e3:.0f} kHz",
                "R_ds_on": f"{u['R_ds_on']['value']*1000:.0f} mohm at {u['T_ref']['value']} degC",
                "R_theta": f"{u['R_theta']['value']} degC/W",
            },
            "source": "Erickson & Maksimovic (2020), Fundamentals of Power Electronics, 3rd ed.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": 5.0})
    print(f"D={float(r['duty_cycle']):.3f}  eta={float(r['efficiency'])*100:.2f}%  T_j={float(r['T_j_degC']):.1f}C")
