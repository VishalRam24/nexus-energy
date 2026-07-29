"""EC167 -- AFE / PFC Rectifier -- F1b Detailed Loss Model -- Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import AFERectifierF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = AFERectifierF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            v_ac_ll_rms  : float or array [V]  AC line-to-line RMS voltage
            v_dc         : float or array [V]  DC bus voltage
            p_input      : float or array [W]  AC input power
            power_factor : float or array      (default 1.0)
        returns:
            efficiency, p_loss_w, modulation_index,
            p_igbt_cond_w, p_igbt_sw_w, p_diode_cond_w, p_diode_rr_w, t_j_degc
        """
        v_ac = np.asarray(inputs["v_ac_ll_rms"], dtype=float)
        v_dc = np.asarray(inputs["v_dc"], dtype=float)
        p_in = np.asarray(inputs["p_input"], dtype=float)
        pf = np.asarray(inputs.get("power_factor", 1.0), dtype=float)

        m = self._model.modulation_index(v_dc, v_ac)
        breakdown = self._model.loss_breakdown(v_ac, v_dc, p_in, pf)
        eta = self._model.efficiency(v_ac, v_dc, p_in, pf)
        p_loss = self._model.total_losses(v_ac, v_dc, p_in, pf)
        t_j = self._model.junction_temperature(v_ac, v_dc, p_in, pf)

        return {
            "efficiency": eta,
            "p_loss_w": p_loss,
            "modulation_index": m,
            "t_j_degc": t_j,
            **breakdown,
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Active Front End (AFE) / PFC Rectifier",
            "ec_id": "EC167",
            "fidelity": "F1b",
            "description": (
                "3-phase AFE/PFC rectifier (VSC topology): "
                "IGBT conduction+switching + freewheeling diode conduction+recovery. "
                "PF = 1.0 achievable via active control."
            ),
            "inputs": {
                "v_ac_ll_rms": {"unit": "V", "range": [200.0, 690.0]},
                "v_dc": {"unit": "V", "range": [400.0, 900.0]},
                "p_input": {"unit": "W", "range": [0.0, 36000.0]},
                "power_factor": {"unit": "dimensionless", "range": [0.7, 1.0]},
            },
            "outputs": {
                "efficiency": {"unit": "dimensionless"},
                "p_loss_w": {"unit": "W"},
                "modulation_index": {"unit": "dimensionless"},
                "t_j_degc": {"unit": "degC"},
                "p_igbt_cond_w": {"unit": "W"},
                "p_igbt_sw_w": {"unit": "W"},
                "p_diode_cond_w": {"unit": "W"},
                "p_diode_rr_w": {"unit": "W"},
            },
            "params": {
                "V_dc": f"{u['V_dc']['value']} V",
                "P_rated": f"{u['P_rated']['value']/1000:.0f} kW",
                "f_sw": f"{u['f_sw']['value']:.0f} Hz",
            },
            "source": "Semikron Application Manual (2015); Mohan et al. (2003)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"v_ac_ll_rms": 400.0, "v_dc": 700.0, "p_input": 25000.0})
    print(f"eta={float(r['efficiency'])*100:.2f}%  m={float(r['modulation_index']):.3f}  "
          f"P_loss={float(r['p_loss_w']):.1f}W  T_j={float(r['t_j_degc']):.1f}°C")
