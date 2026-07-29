"""EC157 — Buck Converter — F1a Ideal Gain + Losses — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import BuckConverterF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = BuckConverterF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            v_in          : float or array [V]   input voltage
            v_out_target  : float or array [V]   desired output voltage
            i_load        : float or array [A]   output load current
        returns:
            duty_cycle      : float or array (dimensionless)
            v_out           : float or array [V]
            efficiency      : float or array (dimensionless)
            p_loss_w        : float or array [W]  total losses
            p_conduction_w  : float or array [W]  conduction losses
            p_switching_w   : float or array [W]  switching losses
        """
        v_in = np.asarray(inputs["v_in"], dtype=float)
        v_out_t = np.asarray(inputs["v_out_target"], dtype=float)
        i_load = np.asarray(inputs["i_load"], dtype=float)

        D = self._model.duty_cycle(v_in, v_out_t)
        v_out = self._model.output_voltage(v_in, v_out_t)
        p_cond = self._model.conduction_losses(v_in, v_out_t, i_load)
        p_sw = self._model.switching_losses(v_in, i_load)
        eta = self._model.efficiency(v_in, v_out_t, i_load)

        return {
            "duty_cycle": D,
            "v_out": v_out,
            "efficiency": eta,
            "p_loss_w": p_cond + p_sw,
            "p_conduction_w": p_cond,
            "p_switching_w": p_sw,
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Buck Converter (Step-Down)",
            "ec_id": "EC157",
            "fidelity": "F1a",
            "description": (
                "V_out=D*V_in; "
                "P_cond=I²*(Rds_on*D+R_L+Vd*(1-D)/V_out); "
                "P_sw=0.5*V_in*I*(t_on+t_off)*f_sw"
            ),
            "inputs": {
                "v_in": {"unit": "V", "range": [20.0, 100.0]},
                "v_out_target": {"unit": "V", "range": [1.0, 50.0]},
                "i_load": {"unit": "A", "range": [0.1, 20.0]},
            },
            "outputs": {
                "duty_cycle": {"unit": "dimensionless"},
                "v_out": {"unit": "V"},
                "efficiency": {"unit": "dimensionless"},
                "p_loss_w": {"unit": "W"},
                "p_conduction_w": {"unit": "W"},
                "p_switching_w": {"unit": "W"},
            },
            "params": {
                "V_in_nominal": f"{u['v_in_nominal']['value']} V",
                "V_out_nominal": f"{u['v_out_nominal']['value']} V",
                "f_sw": f"{u['f_sw']['value']/1e3:.0f} kHz",
                "Rds_on": f"{u['Rds_on']['value']*1000:.0f} mOhm",
                "R_L": f"{u['R_L']['value']*1000:.0f} mOhm",
                "V_diode": f"{u['V_diode']['value']} V",
                "t_on_ns": f"{u['t_on']['value']*1e9:.0f} ns",
                "I_rated": f"{u['I_rated']['value']} A",
            },
            "source": "Erickson & Maksimovic (2020), Fundamentals of Power Electronics, 3rd ed.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": 10.0})
    print(f"D={float(r['duty_cycle']):.3f}  V_out={float(r['v_out']):.2f}V  "
          f"eta={float(r['efficiency'])*100:.2f}%  "
          f"P_cond={float(r['p_conduction_w']):.3f}W  "
          f"P_sw={float(r['p_switching_w']):.3f}W  "
          f"P_loss={float(r['p_loss_w']):.3f}W")
