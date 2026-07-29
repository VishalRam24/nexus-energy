"""EC158 -- Boost Converter -- F1b Detailed Loss Model -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import BoostConverterF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BoostConverterF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            v_in          : float or array [V]
            v_out_target  : float or array [V]
            i_load        : float or array [A]  (output-side current)
        returns:
            duty_cycle, v_out, i_in, efficiency, p_loss_w,
            p_mosfet_cond_w, p_diode_cond_w, p_switching_w, p_inductor_w
        """
        v_in = np.asarray(inputs["v_in"], dtype=float)
        v_out_t = np.asarray(inputs["v_out_target"], dtype=float)
        i_load = np.asarray(inputs["i_load"], dtype=float)

        D = self._model.duty_cycle(v_in, v_out_t)
        v_out = self._model.output_voltage(v_in, v_out_t)
        i_in = self._model.input_current(v_in, v_out_t, i_load)
        breakdown = self._model.loss_breakdown(v_in, v_out_t, i_load)
        eta = self._model.efficiency(v_in, v_out_t, i_load)
        p_loss = self._model.total_losses(v_in, v_out_t, i_load)

        return {
            "duty_cycle": D,
            "v_out": v_out,
            "i_in": i_in,
            "efficiency": eta,
            "p_loss_w": p_loss,
            "p_mosfet_cond_w": breakdown["p_mosfet_cond_w"],
            "p_diode_cond_w": breakdown["p_diode_cond_w"],
            "p_switching_w": breakdown["p_switching_w"],
            "p_inductor_w": breakdown["p_inductor_w"],
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Boost Converter (Step-Up)",
            "ec_id": "EC158",
            "fidelity": "F1b",
            "description": (
                "Detailed semiconductor loss model: "
                "MOSFET conduction (I_in^2*D*R_ds_on), "
                "diode conduction (I_out*V_f), "
                "switching (0.5*V_out*I_in*(t_on+t_off)*f_sw), "
                "inductor DCR (I_in^2*R_L)"
            ),
            "inputs": {
                "v_in": {"unit": "V", "range": [5.0, 40.0]},
                "v_out_target": {"unit": "V", "range": [10.0, 200.0]},
                "i_load": {"unit": "A", "range": [0.0, 15.0]},
            },
            "outputs": {
                "duty_cycle": {"unit": "dimensionless"},
                "v_out": {"unit": "V"},
                "i_in": {"unit": "A"},
                "efficiency": {"unit": "dimensionless"},
                "p_loss_w": {"unit": "W"},
                "p_mosfet_cond_w": {"unit": "W"},
                "p_diode_cond_w": {"unit": "W"},
                "p_switching_w": {"unit": "W"},
                "p_inductor_w": {"unit": "W"},
            },
            "params": {
                "V_in_nominal": f"{u['v_in_nominal']['value']} V",
                "V_out_nominal": f"{u['v_out_nominal']['value']} V",
                "f_sw": f"{u['f_sw']['value']/1e3:.0f} kHz",
                "R_ds_on": f"{u['R_ds_on']['value']*1000:.0f} mohm",
                "V_f": f"{u['V_f']['value']} V",
                "t_on": f"{u['t_on']['value']*1e9:.0f} ns",
                "t_off": f"{u['t_off']['value']*1e9:.0f} ns",
                "R_L": f"{u['R_L']['value']*1000:.0f} mohm",
                "I_rated": f"{u['I_rated']['value']} A",
            },
            "source": "Erickson & Maksimovic (2020), Fundamentals of Power Electronics, 3rd ed.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"v_in": 12.0, "v_out_target": 48.0, "i_load": 5.0})
    print(f"D={float(r['duty_cycle']):.3f}  V_out={float(r['v_out']):.2f}V  "
          f"I_in={float(r['i_in']):.2f}A  eta={float(r['efficiency'])*100:.2f}%")
    print(f"  P_mosfet={float(r['p_mosfet_cond_w']):.3f}W  "
          f"P_diode={float(r['p_diode_cond_w']):.3f}W  "
          f"P_sw={float(r['p_switching_w']):.3f}W  "
          f"P_ind={float(r['p_inductor_w']):.3f}W  "
          f"P_loss={float(r['p_loss_w']):.3f}W")
