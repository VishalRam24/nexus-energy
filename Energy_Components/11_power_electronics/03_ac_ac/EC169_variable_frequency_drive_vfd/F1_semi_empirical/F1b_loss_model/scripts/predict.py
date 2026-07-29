"""EC169 -- VFD -- F1b Detailed Loss Model -- Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import VFDf1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = VFDf1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            p_motor      : float or array [W]   motor shaft power demand
            speed_pu     : float or array [pu]  motor speed per-unit (0–1)
            power_factor : float or array       motor power factor (default 0.85)
        returns:
            efficiency, p_loss_w, p_rectifier_w, p_dc_link_w,
            p_igbt_cond_w, p_igbt_sw_w, p_diode_cond_w, p_diode_rr_w, t_j_degc
        """
        p_motor = np.asarray(inputs["p_motor"], dtype=float)
        speed_pu = np.asarray(inputs["speed_pu"], dtype=float)
        pf = np.asarray(inputs.get("power_factor", 0.85), dtype=float)

        breakdown = self._model.loss_breakdown(p_motor, speed_pu, pf)
        eta = self._model.efficiency(p_motor, speed_pu, pf)
        p_loss = self._model.total_losses(p_motor, speed_pu, pf)
        t_j = self._model.junction_temperature(p_motor, speed_pu, pf)

        return {
            "efficiency": eta,
            "p_loss_w": p_loss,
            "t_j_degc": t_j,
            **breakdown,
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Variable Frequency Drive (VFD)",
            "ec_id": "EC169",
            "fidelity": "F1b",
            "description": (
                "VFD three-stage loss model: "
                "diode bridge rectifier, DC link ESR, "
                "IGBT inverter (conduction + switching + diode recovery). "
                "IEC 61800-9-2 IE1/IE2 class targets."
            ),
            "inputs": {
                "p_motor": {"unit": "W", "range": [0.0, 18000.0]},
                "speed_pu": {"unit": "pu", "range": [0.0, 1.0]},
                "power_factor": {"unit": "dimensionless", "range": [0.6, 1.0]},
            },
            "outputs": {
                "efficiency": {"unit": "dimensionless"},
                "p_loss_w": {"unit": "W"},
                "t_j_degc": {"unit": "degC"},
                "p_rectifier_w": {"unit": "W"},
                "p_dc_link_w": {"unit": "W"},
                "p_igbt_cond_w": {"unit": "W"},
                "p_igbt_sw_w": {"unit": "W"},
                "p_diode_cond_w": {"unit": "W"},
                "p_diode_rr_w": {"unit": "W"},
            },
            "params": {
                "P_rated": f"{u['P_rated']['value']/1000:.0f} kW",
                "V_ac_ll": f"{u['V_ac_ll']['value']} V",
                "f_sw": f"{u['f_sw']['value']:.0f} Hz",
            },
            "source": "IEC 61800-9-2:2017; Semikron Application Manual (2015)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"p_motor": 12000.0, "speed_pu": 0.8})
    print(f"eta={float(r['efficiency'])*100:.2f}%  P_loss={float(r['p_loss_w']):.1f}W  "
          f"T_j={float(r['t_j_degc']):.1f}°C")
    print(f"  P_rect={float(r['p_rectifier_w']):.1f}  P_inv={float(r['p_igbt_cond_w'])+float(r['p_igbt_sw_w']):.1f}W")
