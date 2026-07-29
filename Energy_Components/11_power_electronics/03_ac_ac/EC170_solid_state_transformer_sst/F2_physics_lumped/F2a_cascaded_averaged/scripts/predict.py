"""
EC170 -- Solid State Transformer (SST) -- F2a Cascaded Averaged Three-Stage Model
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import SST_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the SST F2a cascaded averaged model."""

    component_id = "EC170"
    component_name = "Solid State Transformer (SST)"
    fidelity = "F2a -- Cascaded Averaged Three-Stage Model with DC-Link ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = SST_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run the lumped dynamic SST simulation (DC-link + control ODE).

        inputs:
            p_command_w   : float (or callable t->W) commanded through-power.
                            >0 forward (MV->LV), <0 reverse (LV->MV).
            power_factor  : float (default 1.0)
            v_grid_hv_v   : float MV AC RMS input (default 10000) -> voltage transform
            dt            : float (default 0.001 s)
            duration_s    : float (default 0.2 s)
        """
        p_cmd = inputs.get("p_command_w", 8000.0)
        pf = inputs.get("power_factor", 1.0)
        v_hv_ac = inputs.get("v_grid_hv_v", 10000.0)
        dt = inputs.get("dt", 0.001)
        dur = inputs.get("duration_s", 0.2)

        sim = self._model.simulate(p_cmd, power_factor=pf, dt=dt, duration_s=dur)

        # Steady-state cascade summary at the (final / scalar) command
        p_for_summary = p_cmd(dur) if callable(p_cmd) else p_cmd
        casc = self._model.cascade(p_for_summary, power_factor=pf)
        sim["cascade_summary"] = {k: float(v) if v.ndim == 0 else v
                                  for k, v in casc.items()}
        sim["v_lv_ac_transformed_v"] = float(self._model.voltage_transform(v_hv_ac))
        return sim

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "p_command_w": {"unit": "W", "range": [-12000, 12000],
                                "note": "signed; <0 = reverse flow"},
                "power_factor": {"unit": "-", "range": [0.6, 1.0]},
                "v_grid_hv_v": {"unit": "V", "range": [6000, 12000]},
                "dt": {"unit": "s", "range": [1e-4, 1e-2]},
                "duration_s": {"unit": "s", "range": [0.01, 5.0]},
            },
            "outputs": {
                "t": "s",
                "v_hv_dc": "V",
                "v_lv_dc": "V",
                "p_dab_w": "W",
                "p_delivered_w": "W",
                "efficiency": "-",
                "p_loss_w": "W",
                "cascade_summary": "dict (per-stage + total)",
                "v_lv_ac_transformed_v": "V",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"p_command_w": 8000.0, "power_factor": 1.0, "duration_s": 0.1, "dt": 0.001})
    cs = r["cascade_summary"]
    print(f"Final V_hv_dc={r['v_hv_dc'][-1]:.2f} V, V_lv_dc={r['v_lv_dc'][-1]:.2f} V")
    print(f"eta_total={cs['eta_total']:.4f} "
          f"(rect {cs['eta_rect']:.4f} x dab {cs['eta_dab']:.4f} x inv {cs['eta_inv']:.4f}), "
          f"P_delivered={cs['p_delivered_mag_w']:.1f} W, P_loss={cs['p_loss_w']:.1f} W")
    print(f"Voltage transform 10kV -> {r['v_lv_ac_transformed_v']:.1f} V LV AC")
