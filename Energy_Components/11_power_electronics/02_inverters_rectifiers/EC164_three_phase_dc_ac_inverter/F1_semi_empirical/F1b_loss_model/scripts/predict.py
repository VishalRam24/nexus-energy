"""EC164 -- Three-Phase Inverter -- F1b Detailed Loss Model -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import ThreePhaseInverterF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ThreePhaseInverterF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            v_dc          : float [V]  DC bus voltage
            p_load        : float or array [W]  output power
            m             : float  modulation index [0, 1]
            power_factor  : float  load power factor (default 1.0)
        returns:
            v_ac_rms, i_phase_peak, efficiency, p_loss_w,
            p_igbt_cond_w, p_igbt_sw_w, p_diode_cond_w, p_diode_rr_w
        """
        v_dc = float(inputs["v_dc"])
        p_load = np.asarray(inputs["p_load"], dtype=float)
        m = float(inputs.get("m", 0.9))
        pf = float(inputs.get("power_factor", 1.0))

        v_ac = self._model.ac_rms_voltage(v_dc, m)
        i_peak = self._model.phase_peak_current(v_dc, m, p_load, pf)
        bd = self._model.loss_breakdown(v_dc, p_load, m, pf)
        p_loss = self._model.total_losses(v_dc, p_load, m, pf)
        eta = self._model.efficiency(v_dc, p_load, m, pf)

        return {
            "v_ac_rms_V": np.full_like(p_load, v_ac) if np.ndim(p_load) > 0 else v_ac,
            "i_phase_peak_A": i_peak,
            "efficiency": eta,
            "p_loss_w": p_loss,
            "p_igbt_cond_w": bd["p_igbt_cond_w"],
            "p_igbt_sw_w": bd["p_igbt_sw_w"],
            "p_diode_cond_w": bd["p_diode_cond_w"],
            "p_diode_rr_w": bd["p_diode_rr_w"],
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Three-Phase DC-AC Inverter",
            "ec_id": "EC164",
            "fidelity": "F1b",
            "description": (
                "Detailed IGBT/diode loss model: "
                "IGBT conduction (V_ce0*I_avg + r_ce*I_rms^2), "
                "IGBT switching ((E_on+E_off)*f_sw*V/V_ref*I/I_ref), "
                "diode conduction (V_f*I_avg + r_d*I_rms^2), "
                "diode recovery (E_rr*f_sw*V/V_ref), "
                "6 devices total"
            ),
            "inputs": {
                "v_dc": {"unit": "V", "range": [400.0, 1200.0]},
                "p_load": {"unit": "W", "range": [0.0, 120000.0]},
                "m": {"unit": "dimensionless", "range": [0.0, 1.0]},
                "power_factor": {"unit": "dimensionless", "range": [0.0, 1.0]},
            },
            "outputs": {
                "v_ac_rms_V": {"unit": "V"},
                "i_phase_peak_A": {"unit": "A"},
                "efficiency": {"unit": "dimensionless"},
                "p_loss_w": {"unit": "W"},
                "p_igbt_cond_w": {"unit": "W"},
                "p_igbt_sw_w": {"unit": "W"},
                "p_diode_cond_w": {"unit": "W"},
                "p_diode_rr_w": {"unit": "W"},
            },
            "params": {
                "V_dc": f"{u['V_dc']['value']} V",
                "P_rated": f"{u['P_rated']['value']/1e3:.0f} kW",
                "V_ce0": f"{u['V_ce0']['value']} V",
                "r_ce": f"{u['r_ce']['value']*1000:.0f} mohm",
                "E_on": f"{u['E_on']['value']*1000:.0f} mJ",
                "E_off": f"{u['E_off']['value']*1000:.0f} mJ",
                "V_f": f"{u['V_f']['value']} V",
                "r_d": f"{u['r_d']['value']*1000:.0f} mohm",
                "E_rr": f"{u['E_rr']['value']*1000:.0f} mJ",
                "f_sw": f"{u['f_sw']['value']/1e3:.0f} kHz",
            },
            "source": "Semikron Application Manual; Mohan et al. (2003), Power Electronics, 3rd ed.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"v_dc": 800.0, "p_load": 80000.0, "m": 0.9, "power_factor": 0.95})
    print(f"V_ac={float(r['v_ac_rms_V']):.1f}V  I_peak={float(r['i_phase_peak_A']):.1f}A  "
          f"eta={float(r['efficiency'])*100:.2f}%")
    print(f"  P_igbt_cond={float(r['p_igbt_cond_w']):.1f}W  "
          f"P_igbt_sw={float(r['p_igbt_sw_w']):.1f}W  "
          f"P_diode_cond={float(r['p_diode_cond_w']):.1f}W  "
          f"P_diode_rr={float(r['p_diode_rr_w']):.1f}W  "
          f"P_loss={float(r['p_loss_w']):.1f}W")
