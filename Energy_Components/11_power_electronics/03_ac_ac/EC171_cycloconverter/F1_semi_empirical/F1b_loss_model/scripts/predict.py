"""EC171 -- Cycloconverter -- F1b Thyristor Loss Model -- Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import CycloconverterF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CycloconverterF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            p_out         : float or array [W]   output active power
            v_out_ll_rms  : float or array [V]   output line-to-line RMS voltage
            power_factor  : float or array        (default 0.85)
        returns:
            efficiency, p_loss_w, p_conduction_w, p_snubber_w,
            firing_angle_deg, t_j_degc
        """
        p_out = np.asarray(inputs["p_out"], dtype=float)
        v_out = np.asarray(inputs["v_out_ll_rms"], dtype=float)
        pf = np.asarray(inputs.get("power_factor", 0.85), dtype=float)

        breakdown = self._model.loss_breakdown(p_out, v_out, pf)
        eta = self._model.efficiency(p_out, v_out, pf)
        p_loss = self._model.total_losses(p_out, v_out, pf)
        t_j = self._model.junction_temperature(p_out, v_out, pf)
        alpha = self._model.firing_angle(v_out)

        return {
            "efficiency": eta,
            "p_loss_w": p_loss,
            "firing_angle_deg": np.degrees(alpha),
            "t_j_degc": t_j,
            **breakdown,
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Cycloconverter",
            "ec_id": "EC171",
            "fidelity": "F1b",
            "description": (
                "SCR cycloconverter (direct AC-AC): "
                "per-thyristor conduction (V_T0*I_avg + r_T*I_rms^2) "
                "and R-C snubber dissipation. "
                "No DC link: commutation at line frequency."
            ),
            "inputs": {
                "p_out": {"unit": "W", "range": [0.0, 600000.0]},
                "v_out_ll_rms": {"unit": "V", "range": [50.0, 650.0]},
                "power_factor": {"unit": "dimensionless", "range": [0.5, 1.0]},
            },
            "outputs": {
                "efficiency": {"unit": "dimensionless"},
                "p_loss_w": {"unit": "W"},
                "firing_angle_deg": {"unit": "deg"},
                "t_j_degc": {"unit": "degC"},
                "p_conduction_w": {"unit": "W"},
                "p_snubber_w": {"unit": "W"},
            },
            "params": {
                "P_rated": f"{u['P_rated']['value']/1000:.0f} kW",
                "n_SCR": str(2 * 3 * int(u["n_phase_out"]["value"])),
                "V_in_ll": f"{u['V_in_ll']['value']} V",
                "f_line": f"{u['f_line']['value']:.0f} Hz",
            },
            "source": "Mohan, Undeland & Robbins (2003); Rashid (2011)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"p_out": 400000.0, "v_out_ll_rms": 500.0})
    print(f"eta={float(r['efficiency'])*100:.2f}%  "
          f"alpha={float(r['firing_angle_deg']):.1f}°  "
          f"P_loss={float(r['p_loss_w']):.0f}W  T_j={float(r['t_j_degc']):.1f}°C")
