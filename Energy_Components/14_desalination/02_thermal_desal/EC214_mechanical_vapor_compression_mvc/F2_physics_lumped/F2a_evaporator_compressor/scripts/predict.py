"""
EC214 -- Mechanical Vapor Compression (MVC) -- F2a Physics-Lumped
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import MVC_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the MVC F2a physics-lumped evaporator model."""

    component_id = "EC214"
    component_name = "Mechanical Vapor Compression (MVC) Desalination"
    fidelity = "F2a -- Physics-Lumped Single-Effect Evaporator + Vapor Compressor"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = MVC_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a steady design-point evaluation plus the lumped transient.

        inputs (all optional, fall back to data/parameters.json):
            T_brine_C    : float  evaporator boiling temperature [degC]
            dT_lift_C    : float  compressor temperature lift [K]
            T0_brine_C   : float  initial brine temp for the transient [degC]
            duration_s   : float  simulation horizon [s]   (default 3000)
            dt           : float  output step [s]          (default 10)

        returns dict with steady design metrics + transient time series.
        """
        Tb = inputs.get("T_brine_C", None)
        dT = inputs.get("dT_lift_C", None)
        if Tb is not None:
            self._model.T_brine_C = Tb
        if dT is not None:
            self._model.dT_lift = dT

        T0 = inputs.get("T0_brine_C", None)
        dur = inputs.get("duration_s", 3000.0)
        dt = inputs.get("dt", 10.0)

        design = self._model.design_point()
        sim = self._model.simulate(T0_brine_C=T0, duration_s=dur, dt=dt)

        out = dict(design)
        out.update({
            "t": sim["t"],
            "T_brine_C_series": sim["T_brine_C"],
            "level_m_series": sim["level_m"],
            "distillate_m3_day_series": sim["distillate_m3_day"],
            "P_elec_kW_series": sim["P_elec_kW"],
            "SEC_kWh_m3_series": sim["SEC_kWh_m3"],
            "SEC_kWh_m3_final": float(sim["SEC_kWh_m3"][-1]),
            "distillate_m3_day_final": float(sim["distillate_m3_day"][-1]),
            "success": sim["success"],
        })
        return out

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_brine_C": {"unit": "degC", "range": [50, 75]},
                "dT_lift_C": {"unit": "K", "range": [2, 8]},
                "T0_brine_C": {"unit": "degC", "range": [20, 75]},
                "duration_s": {"unit": "s"},
                "dt": {"unit": "s"},
            },
            "outputs": {
                "SEC_kWh_m3": "kWh/m3 (specific electric energy)",
                "GOR_equiv": "- (heat-pump amplification hfg/w_compressor)",
                "P_elec_kW": "kW (compressor electrical power)",
                "distillate_m3_day": "m3/day",
                "T_steam_C": "degC (compressor discharge sat. temp)",
                "BPE_C": "K (boiling-point elevation)",
                "pressure_ratio": "-",
                "t / *_series": "transient time series arrays",
            },
            "source": self._raw.get("unit", {}).get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"duration_s": 2000.0, "dt": 20.0})
    print(f"SEC = {r['SEC_kWh_m3']:.2f} kWh/m3 | "
          f"distillate = {r['distillate_m3_day_final']:.1f} m3/day | "
          f"T_steam = {r['T_steam_C']:.2f} C | BPE = {r['BPE_C']:.3f} K | "
          f"GOR_eq = {r['GOR_equiv']:.1f} | PR = {r['pressure_ratio']:.3f}")
