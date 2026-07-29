"""EC174 -- Instrument Transformer (CT/PT) -- F1a Ideal Ratio -- Predict Interface"""
import json, sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import InstrumentTransformerF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = InstrumentTransformerF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            i_primary           : float or array [A]   CT primary current
            v_primary           : float or array [V]   PT primary voltage
            current_fraction    : float or array [-]   I_pri / I_rated (for accuracy check)
        returns:
            ct_i_secondary      : float or array [A]   CT secondary current
            pt_v_secondary      : float or array [V]   PT secondary voltage
            ct_burden_va        : float or array [VA]  CT secondary burden power
            pt_burden_va        : float or array [VA]  PT secondary burden power
            ratio_error_limit_pct : float              max ratio error (IEC Class 0.2)
            phase_error_limit_min : float              max phase error [arcmin]
            within_accuracy_class : bool or array      True if within Class 0.2 range
        """
        i_pri = np.asarray(inputs.get("i_primary", self._model.I_rated_pri), dtype=float)
        v_pri = np.asarray(inputs.get("v_primary", self._model.V_rated_pri), dtype=float)
        cf = np.asarray(inputs.get("current_fraction", i_pri / self._model.I_rated_pri), dtype=float)

        return {
            "ct_i_secondary": self._model.ct_secondary_current(i_pri),
            "pt_v_secondary": self._model.pt_secondary_voltage(v_pri),
            "ct_burden_va": self._model.ct_burden_power(i_pri),
            "pt_burden_va": self._model.pt_burden_power(v_pri),
            "ratio_error_limit_pct": np.full_like(cf, self._model.ratio_error_pct),
            "phase_error_limit_min": np.full_like(cf, self._model.phase_error_min),
            "within_accuracy_class": self._model.accuracy_within_class(cf),
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        m = self._model
        return {
            "name": "Instrument Transformer (CT/PT)",
            "ec_id": "EC174",
            "fidelity": "F1a",
            "description": "I_sec=I_pri/N_ct; V_sec=V_pri*N_pt; IEC Class 0.2; ratio_err<0.1%",
            "inputs": {
                "i_primary": {"unit": "A", "range": [0.0, 2000.0]},
                "v_primary": {"unit": "V", "range": [0.0, 15000.0]},
                "current_fraction": {"unit": "dimensionless", "range": [0.01, 2.0], "optional": True},
            },
            "outputs": {
                "ct_i_secondary": {"unit": "A"},
                "pt_v_secondary": {"unit": "V"},
                "ct_burden_va": {"unit": "VA"},
                "pt_burden_va": {"unit": "VA"},
                "ratio_error_limit_pct": {"unit": "%"},
                "phase_error_limit_min": {"unit": "arcmin"},
                "within_accuracy_class": {"unit": "bool"},
            },
            "params": {
                "CT_ratio": f"{m.I_rated_pri:.0f}A/{m.I_rated_sec:.0f}A",
                "PT_ratio": f"{m.V_rated_pri:.0f}V/{m.V_rated_sec:.0f}V",
                "accuracy_class": u["accuracy_class"]["value"],
                "ratio_error_pct": m.ratio_error_pct,
                "phase_error_arcmin": m.phase_error_min,
            },
            "source": "IEC 61869-2:2012; IEC 61869-3:2011",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for i_pri in [100.0, 500.0, 1000.0, 1500.0]:
        cf = i_pri / model._model.I_rated_pri
        r = model.predict({"i_primary": i_pri, "v_primary": 11000.0, "current_fraction": cf})
        print(f"I_pri={i_pri:.0f}A: I_sec={float(r['ct_i_secondary']):.2f}A  "
              f"V_sec={float(r['pt_v_secondary']):.1f}V  "
              f"Burden={float(r['ct_burden_va']):.2f}VA  "
              f"InClass={bool(r['within_accuracy_class'])}")
