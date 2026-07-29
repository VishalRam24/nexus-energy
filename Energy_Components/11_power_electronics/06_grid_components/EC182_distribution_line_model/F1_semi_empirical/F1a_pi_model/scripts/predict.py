"""EC182 — Distribution Line — F1a R+jX Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import DistributionLinePiModel


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = DistributionLinePiModel(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            V_s_kV      : float or array [kV]   sending-end line-to-line voltage
            P_load_kW   : float or array [kW]   active load at receiving end
            Q_load_kVAR : float or array [kVAR] reactive load (+ = inductive)
            length_km   : float (optional)      feeder length in km
        returns:
            V_r_kV          : [kV]    receiving-end voltage (L-L)
            I_line_A        : [A]     line current magnitude
            P_loss_kW       : [kW]    active power loss (3-phase)
            Q_loss_kVAR     : [kVAR]  reactive power loss
            P_s_kW          : [kW]    sending-end active power
            efficiency      : [—]     P_load / P_s
            voltage_drop_kV : [kV]    |V_s| - |V_r|
            voltage_drop_pct: [%]     voltage drop as % of V_s
            power_factor_load: [—]    load power factor
        """
        length_km = inputs.get("length_km", None)
        return self._model.compute(
            V_s_kV=inputs["V_s_kV"],
            P_load_kW=inputs["P_load_kW"],
            Q_load_kVAR=inputs["Q_load_kVAR"],
            length_km=length_km,
        )

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Distribution Line (Feeder, R+jX Model)",
            "ec_id": "EC182",
            "fidelity": "F1a",
            "description": "Series R+jX model; P_loss=3*I^2*R; iterative V_r solve",
            "inputs": {
                "V_s_kV": {"unit": "kV", "range": [4.0, 35.0]},
                "P_load_kW": {"unit": "kW", "range": [0.0, 5000.0]},
                "Q_load_kVAR": {"unit": "kVAR", "range": [-2000.0, 2000.0]},
                "length_km": {"unit": "km", "range": [0.1, 50.0], "optional": True},
            },
            "outputs": {
                "V_r_kV": {"unit": "kV"},
                "I_line_A": {"unit": "A"},
                "P_loss_kW": {"unit": "kW"},
                "Q_loss_kVAR": {"unit": "kVAR"},
                "P_s_kW": {"unit": "kW"},
                "efficiency": {"unit": "dimensionless"},
                "voltage_drop_kV": {"unit": "kV"},
                "voltage_drop_pct": {"unit": "%"},
                "power_factor_load": {"unit": "dimensionless"},
            },
            "params": {
                "V_base": f"{u['V_base_kV']['value']} kV",
                "R_ohm_per_km": u["R_ohm_per_km"]["value"],
                "X_ohm_per_km": u["X_ohm_per_km"]["value"],
                "default_length_km": u["length_km"]["value"],
            },
            "source": "Kersting (2012), Distribution System Modeling and Analysis, 3rd ed.",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"V_s_kV": 11.0, "P_load_kW": 1500.0, "Q_load_kVAR": 600.0})
    print(f"V_r={float(r['V_r_kV']):.3f} kV  "
          f"P_loss={float(r['P_loss_kW']):.2f} kW  "
          f"dV={float(r['voltage_drop_pct']):.2f}%  "
          f"eta={float(r['efficiency'])*100:.2f}%")
