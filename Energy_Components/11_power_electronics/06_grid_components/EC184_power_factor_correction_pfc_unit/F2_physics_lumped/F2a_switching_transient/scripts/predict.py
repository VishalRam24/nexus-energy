"""
EC184 -- Power Factor Correction Unit -- F2a Physics-Lumped (Switching Transient)
Standardised predict() / get_info() interface.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import PFCUnit_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC184 PFC F2a physics-lumped model."""

    component_id = "EC184"
    component_name = "Power Factor Correction Unit (Shunt Capacitor Bank)"
    fidelity = "F2a -- Physics-Lumped: reactive compensation + resonance + RLC inrush ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = PFCUnit_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Steady-state compensation + resonance + energization inrush.

        inputs:
            P_kW            : active load [kW]                  (default 800)
            pf_initial      : initial lagging PF               (default 0.80)
            pf_target       : desired PF                       (default param)
            Q_comp_kVAR     : override delivered Qc [kVAR]      (optional)
            Lsys_mH         : source inductance [mH]            (optional)
            detuning_pct    : series detuning reactor p%        (default 0)
            switch_angle_deg: point-on-wave closing [deg]       (default 90)
            duration_s      : energization sim length [s]       (default 0.06)
            simulate_inrush : bool, run the ODE                 (default True)
        """
        P = inputs.get("P_kW", 800.0)
        pf1 = inputs.get("pf_initial", 0.80)
        pf2 = inputs.get("pf_target", None)
        Qc_override = inputs.get("Q_comp_kVAR", None)
        Lsys_mH = inputs.get("Lsys_mH", None)
        Lsys_H = Lsys_mH * 1e-3 if Lsys_mH is not None else None
        detuning_pct = inputs.get("detuning_pct", 0.0)
        switch_angle = inputs.get("switch_angle_deg", 90.0)
        duration = inputs.get("duration_s", 0.06)
        do_inrush = inputs.get("simulate_inrush", True)

        comp = self._model.compensate(P, pf1, pf2, Qc_override)
        Qc = comp["Q_compensated_kVAR"]
        res = self._model.resonance(Qc, Lsys_H, detuning_pct if detuning_pct > 0 else None)
        dV = self._model.voltage_rise(Qc, Lsys_H)

        out = dict(comp)
        out["resonance"] = res
        out["voltage_rise_pu"] = dV
        out["C_F"] = self._model.capacitance_for_Q(Qc)

        if do_inrush and Qc > 0:
            tr = self._model.energize(
                Qc_kVAR=Qc, Lsys_H=Lsys_H, detuning_pct=detuning_pct,
                switch_angle_deg=switch_angle, duration_s=duration,
            )
            out["transient"] = tr

        return out

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "P_kW": {"unit": "kW", "range": [0, 10000]},
                "pf_initial": {"unit": "-", "range": [0.5, 0.99]},
                "pf_target": {"unit": "-", "range": [0.8, 1.0]},
                "Q_comp_kVAR": {"unit": "kVAR", "range": [0, 2000]},
                "Lsys_mH": {"unit": "mH", "range": [0.1, 200]},
                "detuning_pct": {"unit": "%", "range": [0, 14]},
                "switch_angle_deg": {"unit": "deg", "range": [0, 360]},
                "duration_s": {"unit": "s", "range": [0.001, 1.0]},
            },
            "outputs": {
                "Q_compensated_kVAR": "kVAR",
                "pf_achieved": "-",
                "released_capacity_kVA": "kVA",
                "voltage_rise_pu": "-",
                "resonance": "dict (h_parallel, h_tune, Ssc_MVA)",
                "transient": "dict time-series (i, v_cap) + inrush_factor",
            },
            "source": self._raw.get("unit", {}).get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"P_kW": 800.0, "pf_initial": 0.80, "pf_target": 0.95})
    print(f"\nQc delivered   : {r['Q_compensated_kVAR']:.1f} kVAR")
    print(f"PF: {r['pf_initial']:.3f} -> {r['pf_achieved']:.3f}")
    print(f"Released capacity: {r['released_capacity_kVA']:.1f} kVA")
    print(f"Voltage rise   : {r['voltage_rise_pu']*100:.2f} %")
    print(f"Parallel resonance h = {r['resonance']['h_parallel']:.2f} "
          f"({r['resonance']['f_parallel_Hz']:.0f} Hz)")
    print(f"Inrush factor  : {r['transient']['inrush_factor']:.1f}x  "
          f"(I_peak={r['transient']['I_peak_A']:.0f} A, "
          f"f_nat={r['transient']['f_natural_Hz']:.0f} Hz)")
