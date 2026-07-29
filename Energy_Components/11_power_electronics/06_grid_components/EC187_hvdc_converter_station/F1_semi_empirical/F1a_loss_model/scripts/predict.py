"""EC187 — HVDC Converter Station — F1a Loss Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import HVDCConverterModel


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = HVDCConverterModel(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            P_transfer_MW   : float or array [MW]   power transfer (0 to P_rated)
            direction       : str   "rectifier" or "inverter" (default "rectifier")
            Q_request_MVAR  : float (optional)      reactive support request [MVAR]
        returns:
            P_in_MW         : [MW]  input power
            P_out_MW        : [MW]  output power
            P_loss_MW       : [MW]  station losses
            efficiency      : [—]   P_out / P_in
            I_dc_kA         : [kA]  DC current
            utilization     : [—]   P_transfer / P_rated
            Q_delivered_MVAR: [MVAR] reactive output (VSC capability)
            direction       : str
        """
        return self._model.compute(
            P_transfer_MW=inputs["P_transfer_MW"],
            direction=inputs.get("direction", "rectifier"),
            Q_request_MVAR=inputs.get("Q_request_MVAR", 0.0),
        )

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "HVDC Converter Station (VSC)",
            "ec_id": "EC187",
            "fidelity": "F1a",
            "description": "P_loss=P_no_load+loss_factor*P; eta=P_out/P_in; I_dc=P/(V_dc)",
            "inputs": {
                "P_transfer_MW": {"unit": "MW", "range": [0.0, 1000.0]},
                "direction": {"unit": "enum", "values": ["rectifier", "inverter"]},
                "Q_request_MVAR": {"unit": "MVAR", "range": [-400.0, 400.0], "optional": True},
            },
            "outputs": {
                "P_in_MW": {"unit": "MW"},
                "P_out_MW": {"unit": "MW"},
                "P_loss_MW": {"unit": "MW"},
                "efficiency": {"unit": "dimensionless"},
                "I_dc_kA": {"unit": "kA"},
                "utilization": {"unit": "dimensionless"},
                "Q_delivered_MVAR": {"unit": "MVAR"},
            },
            "params": {
                "P_rated": f"{u['P_rated_MW']['value']} MW",
                "V_dc": f"±{u['V_dc_kV']['value']} kV",
                "loss_factor": f"{u['loss_factor_station']['value']*100:.1f}% per station",
                "P_no_load": f"{u['P_no_load_MW']['value']} MW",
                "type": u["type"]["value"],
            },
            "source": "Cigre TB 492 (2012), VSC Transmission; Alstom (2010) HVDC Reference",
        }


if __name__ == "__main__":
    model = ComponentModel()
    for P in [100, 300, 600, 800, 1000]:
        r = model.predict({"P_transfer_MW": float(P), "direction": "rectifier"})
        print(f"P={P:>5} MW  P_loss={float(r['P_loss_MW']):.2f} MW  "
              f"eta={float(r['efficiency'])*100:.2f}%  I_dc={float(r['I_dc_kA']):.3f} kA")
