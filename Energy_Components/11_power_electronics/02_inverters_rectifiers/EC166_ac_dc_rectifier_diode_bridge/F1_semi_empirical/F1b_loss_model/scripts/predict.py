"""EC166 -- AC-DC Diode Bridge Rectifier -- F1b Detailed Loss Model -- Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import DiodeBridgeRectifierF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = DiodeBridgeRectifierF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            v_ac_rms : float or array [V]  AC supply line-to-line RMS voltage
            i_dc     : float or array [A]  DC output current
        returns:
            v_dc, efficiency, p_loss_w, p_conduction_w, p_recovery_w, t_j_degc
        """
        v_ac = np.asarray(inputs["v_ac_rms"], dtype=float)
        i_dc = np.asarray(inputs["i_dc"], dtype=float)

        v_dc = self._model.dc_voltage(v_ac)
        breakdown = self._model.loss_breakdown(v_ac, i_dc)
        eta = self._model.efficiency(v_ac, i_dc)
        p_loss = self._model.total_losses(v_ac, i_dc)
        t_j = self._model.junction_temperature(i_dc, v_ac)

        return {
            "v_dc": v_dc,
            "efficiency": eta,
            "p_loss_w": p_loss,
            "t_j_degc": t_j,
            **breakdown,
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "AC-DC Rectifier (Diode Bridge)",
            "ec_id": "EC166",
            "fidelity": "F1b",
            "description": (
                "Per-diode loss model: conduction (V_f*I_avg + r_d*I_rms^2) "
                "and reverse recovery at line frequency."
            ),
            "inputs": {
                "v_ac_rms": {"unit": "V (L-L RMS)", "range": [100.0, 690.0]},
                "i_dc": {"unit": "A", "range": [0.0, 150.0]},
            },
            "outputs": {
                "v_dc": {"unit": "V"},
                "efficiency": {"unit": "dimensionless"},
                "p_loss_w": {"unit": "W"},
                "p_conduction_w": {"unit": "W"},
                "p_recovery_w": {"unit": "W"},
                "t_j_degc": {"unit": "degC"},
            },
            "params": {
                "n_phases": str(int(u["n_phases"]["value"])),
                "V_f": f"{u['V_f']['value']} V",
                "r_d": f"{u['r_d']['value']*1000:.1f} mohm",
                "f_line": f"{u['f_line']['value']:.0f} Hz",
            },
            "source": "Mohan, Undeland & Robbins (2003), Power Electronics, 3rd ed.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"v_ac_rms": 400.0, "i_dc": 30.0})
    print(f"V_dc={float(r['v_dc']):.2f}V  eta={float(r['efficiency'])*100:.2f}%  "
          f"P_loss={float(r['p_loss_w']):.2f}W  T_j={float(r['t_j_degc']):.1f}°C")
