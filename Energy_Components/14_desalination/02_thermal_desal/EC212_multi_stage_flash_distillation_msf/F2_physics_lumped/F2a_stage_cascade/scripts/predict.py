"""
EC212 -- Multi-Stage Flash Distillation (MSF) -- F2a Stage-Cascade (physics-lumped)
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import MSF_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for MSF F2a stage-cascade physics-lumped model."""

    component_id = "EC212"
    component_name = "Multi-Stage Flash Distillation (MSF)"
    fidelity = "F2a -- Stage-Cascade Mass/Energy Balance + Lumped Transient ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = MSF_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run MSF steady-state + transient simulation.

        inputs:
            T_top_brine_C : float  top brine temperature / TBT (default param value)
            T_last_stage_C: float  coldest-stage brine temp (default param value)
            M_recirc_kg_s : float  recirculating brine mass flow (default param value)
            T0_C          : float  initial stage temperature for transient (default seawater)
            duration_s    : float  transient duration (default 1500 s)
            n_eval        : int    number of time samples (default 200)

        returns dict with steady-state metrics (GOR, distillate, performance ratio,
        temperature cascade, steam, recovery) and the transient stage-temperature arrays.
        """
        Tt = inputs.get("T_top_brine_C", None)
        Tl = inputs.get("T_last_stage_C", None)
        Mb = inputs.get("M_recirc_kg_s", None)
        T0 = inputs.get("T0_C", None)
        dur = inputs.get("duration_s", 1500.0)
        n_eval = int(inputs.get("n_eval", 200))

        result = self._model.simulate(T_top=Tt, T_last=Tl, M_brine=Mb,
                                      T0=T0, duration_s=dur, n_eval=n_eval)
        return result

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_top_brine_C": {"unit": "degC", "range": [90, 120]},
                "T_last_stage_C": {"unit": "degC", "range": [30, 50]},
                "M_recirc_kg_s": {"unit": "kg/s", "range": [100, 5000]},
                "T0_C": {"unit": "degC"},
                "duration_s": {"unit": "s"},
                "n_eval": {"unit": "-"},
            },
            "outputs": {
                "t": "s",
                "T_stages": "degC (n_eval x N_stages) transient cascade",
                "T_target": "degC (N_stages,) steady cascade",
                "T_stage": "degC (N_stages,) stage saturation temps",
                "distillate_stage": "kg/s per stage",
                "D_total": "kg/s total distillate",
                "M_steam": "kg/s brine-heater steam",
                "Q_heater": "kW brine-heater duty",
                "GOR": "- gain output ratio (kg distillate / kg steam)",
                "PR": "- performance ratio (kg distillate / 2326 kJ)",
                "flash_range": "degC (TBT - T_last)",
                "NEA": "degC non-equilibrium allowance per stage",
                "recovery": "- distillate / recirc brine",
                "salinity_out_ppm": "ppm blowdown salinity",
            },
            "references": [
                "El-Dessouky, H.T. & Ettouney, H.M. (2002). Fundamentals of Salt Water "
                "Desalination. Elsevier, Ch. 7.",
                "Khawaji, Kutubkhanah & Wie (2008). Desalination 221:47-69.",
            ],
            "source": self._raw["unit"].get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    info = m.get_info()
    print(f"{info['component_id']} {info['component_name']} [{info['fidelity']}]")
    r = m.predict({"duration_s": 1500.0, "n_eval": 100})
    print(f"GOR            : {r['GOR']:.2f}")
    print(f"Performance PR : {r['PR']:.2f}")
    print(f"Distillate     : {r['D_total']:.2f} kg/s  ({r['D_total']*3.6:.0f} m3/h)")
    print(f"Steam demand   : {r['M_steam']:.2f} kg/s")
    print(f"Flash range    : {r['flash_range']:.1f} degC over {len(r['T_stage'])} stages")
    print(f"NEA / stage    : {r['NEA']:.3f} degC")
    print(f"Recovery       : {r['recovery']*100:.1f} %  (blowdown {r['salinity_out_ppm']:.0f} ppm)")
    print(f"Cascade top/bot: {r['T_target'][0]:.1f} -> {r['T_target'][-1]:.1f} degC")
    print(f"Transient end  : stage0={r['T_stages'][-1,0]:.1f}, "
          f"stageN={r['T_stages'][-1,-1]:.1f} degC (target {r['T_target'][-1]:.1f})")
