"""EC174 -- Instrument Transformer (CT/PT) -- F1b Accuracy + Loss Model -- Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import InstrumentTransformerF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = InstrumentTransformerF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            input_value  : float or array
                          For CT: primary current [A]
                          For PT: primary voltage [V]
        returns:
            p_loss_w, p_copper_w, p_core_w,
            ratio_error_pct, within_accuracy_class, t_winding_degc
        """
        x = np.asarray(inputs["input_value"], dtype=float)

        breakdown = self._model.loss_breakdown(x)
        p_loss = self._model.total_losses(x)
        err = self._model.accuracy_error_pct(x)
        ok = self._model.within_accuracy_class(x)
        t_wind = self._model.junction_temperature(x)

        return {
            "p_loss_w": p_loss,
            "p_copper_w": breakdown["p_copper_w"],
            "p_core_w": breakdown["p_core_w"],
            "ratio_error_pct": err,
            "within_accuracy_class": ok,
            "t_winding_degc": t_wind,
        }

    def get_info(self) -> dict:
        u = self.params["unit"]
        device_type = u["type"]["value"]
        return {
            "name": f"Instrument Transformer ({device_type})",
            "ec_id": "EC174",
            "fidelity": "F1b",
            "description": (
                f"{'Current' if device_type == 'CT' else 'Voltage'} transformer "
                f"accuracy + loss model. "
                f"Ratio error, copper loss, core loss, and thermal. "
                f"IEC 60044 accuracy class {u['accuracy_class']['value']}."
            ),
            "inputs": {
                "input_value": {
                    "unit": "A" if device_type == "CT" else "V",
                    "description": "Primary current (CT) or voltage (PT)",
                },
            },
            "outputs": {
                "p_loss_w": {"unit": "W"},
                "p_copper_w": {"unit": "W"},
                "p_core_w": {"unit": "W"},
                "ratio_error_pct": {"unit": "%"},
                "within_accuracy_class": {"unit": "bool"},
                "t_winding_degc": {"unit": "degC"},
            },
            "params": {
                "type": device_type,
                "turns_ratio": str(u["turns_ratio"]["value"]),
                "accuracy_class": str(u["accuracy_class"]["value"]),
                "burden": f"{u['S_burden_VA']['value']} VA at PF {u['pf_burden']['value']}",
            },
            "source": "IEC 60044-1:1996 / IEC 60044-2:1997",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    # CT at rated primary current
    r = model.predict({"input_value": 200.0})
    print(f"CT @ 200A: P_loss={float(r['p_loss_w']):.3f}W  "
          f"Ratio_err={float(r['ratio_error_pct']):.4f}%  "
          f"OK={bool(r['within_accuracy_class'])}  "
          f"T_wind={float(r['t_winding_degc']):.1f}°C")
    # CT at 10% primary current (accuracy specification point)
    r2 = model.predict({"input_value": 20.0})
    print(f"CT @ 20A (10%):  Ratio_err={float(r2['ratio_error_pct']):.4f}%  "
          f"OK={bool(r2['within_accuracy_class'])}")
