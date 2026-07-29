"""EC164 -- Three-Phase Inverter -- F2a dq-Frame -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import ThreePhaseInverterF2a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = ThreePhaseInverterF2a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            P_ref_kw, Q_ref_kvar, dt, duration_s
        returns:
            t, i_d, i_q, P, Q, v_dc
        """
        return self._model.simulate(
            inputs["P_ref_kw"],
            inputs["Q_ref_kvar"],
            inputs.get("dt", 1e-5),
            inputs.get("duration_s", 0.1),
            x0=inputs.get("x0", None),
        )

    def predict_steady_state(self, inputs: dict) -> dict:
        return self._model.steady_state(inputs["P_ref_kw"], inputs["Q_ref_kvar"])

    def get_info(self) -> dict:
        return {
            "name": "Three-Phase DC-AC Inverter",
            "ec_id": "EC164",
            "fidelity": "F2a",
            "sub_fidelity": "dq_frame",
            "description": (
                "dq-frame averaged model with PI current control. "
                "di_d/dt = (v_d - R*i_d + w*L*i_q - e_d)/L, "
                "di_q/dt = (v_q - R*i_q - w*L*i_d - e_q)/L"
            ),
            "inputs": {
                "P_ref_kw": {"unit": "kW", "range": [-120, 120]},
                "Q_ref_kvar": {"unit": "kvar", "range": [-60, 60]},
                "dt": {"unit": "s", "default": 1e-5},
                "duration_s": {"unit": "s", "default": 0.1},
            },
            "outputs": {
                "t": {"unit": "s"},
                "i_d": {"unit": "A"},
                "i_q": {"unit": "A"},
                "P": {"unit": "W"},
                "Q": {"unit": "var"},
                "v_dc": {"unit": "V"},
            },
            "source": "Teodorescu et al. (2011), Grid Converters for PV and Wind. Wiley.",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    ss = model.predict_steady_state({"P_ref_kw": 50.0, "Q_ref_kvar": 0.0})
    r = model.predict({
        "P_ref_kw": 50.0, "Q_ref_kvar": 0.0,
        "dt": 1e-5, "duration_s": 0.1,
    })
    print(f"SS: i_d={ss['i_d_ss']:.2f}A  i_q={ss['i_q_ss']:.2f}A  P={ss['P_ss_w']:.0f}W")
    print(f"Sim final: i_d={r['i_d'][-1]:.2f}A  P={r['P'][-1]:.0f}W")
