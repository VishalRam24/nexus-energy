"""EC165 -- Multilevel Inverter -- F1b Detailed Loss Model -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import MultilevelInverterF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MultilevelInverterF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            v_dc         : float or array [V]  DC bus voltage
            p_load       : float or array [W]  output active power
            m            : float or array      modulation index [0.1..1.15]
            power_factor : float or array      load power factor [0.6..1.0] (default 0.9)
        returns:
            efficiency, p_loss_w, p_outer_igbt_cond_w, p_inner_igbt_cond_w,
            p_outer_igbt_sw_w, p_inner_igbt_sw_w, p_clamp_diode_cond_w,
            p_clamp_diode_rr_w, t_j_degc
        """
        v_dc = np.asarray(inputs["v_dc"], dtype=float)
        p_load = np.asarray(inputs["p_load"], dtype=float)
        m = np.asarray(inputs["m"], dtype=float)
        pf = np.asarray(inputs.get("power_factor", 0.9), dtype=float)

        breakdown = self._model.loss_breakdown(v_dc, p_load, m, pf)
        eta = self._model.efficiency(v_dc, p_load, m, pf)
        p_loss = self._model.total_losses(v_dc, p_load, m, pf)
        t_j = self._model.junction_temperature(v_dc, p_load, m, pf)

        return {
            "efficiency": eta,
            "p_loss_w": p_loss,
            "t_j_degc": t_j,
            **breakdown,
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Multilevel Inverter (3-level NPC)",
            "ec_id": "EC165",
            "fidelity": "F1b",
            "description": (
                "3-level NPC inverter per-device loss model: "
                "outer IGBT conduction/switching, inner IGBT conduction/switching, "
                "clamping diode conduction + reverse recovery."
            ),
            "inputs": {
                "v_dc": {"unit": "V", "range": [300.0, 900.0]},
                "p_load": {"unit": "W", "range": [0.0, 60000.0]},
                "m": {"unit": "dimensionless", "range": [0.1, 1.15]},
                "power_factor": {"unit": "dimensionless", "range": [0.6, 1.0]},
            },
            "outputs": {
                "efficiency": {"unit": "dimensionless"},
                "p_loss_w": {"unit": "W"},
                "t_j_degc": {"unit": "degC"},
                "p_outer_igbt_cond_w": {"unit": "W"},
                "p_inner_igbt_cond_w": {"unit": "W"},
                "p_outer_igbt_sw_w": {"unit": "W"},
                "p_inner_igbt_sw_w": {"unit": "W"},
                "p_clamp_diode_cond_w": {"unit": "W"},
                "p_clamp_diode_rr_w": {"unit": "W"},
            },
            "params": {
                "V_dc": f"{u['V_dc']['value']} V",
                "P_rated": f"{u['P_rated']['value']/1000:.0f} kW",
                "f_sw": f"{u['f_sw']['value']:.0f} Hz",
                "n_levels": str(int(u["n_levels"]["value"])),
            },
            "source": "Semikron Application Manual (2015); Nabae et al. (1981)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"v_dc": 700.0, "p_load": 40000.0, "m": 0.9, "power_factor": 0.9})
    print(f"eta={float(r['efficiency'])*100:.2f}%  P_loss={float(r['p_loss_w']):.1f}W  "
          f"T_j={float(r['t_j_degc']):.1f}°C")
