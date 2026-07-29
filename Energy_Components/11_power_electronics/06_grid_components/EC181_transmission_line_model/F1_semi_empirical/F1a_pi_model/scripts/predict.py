"""EC181 — Transmission Line — F1a Pi-Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import TransmissionLinePiModel


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = TransmissionLinePiModel(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            V_s_pu      : float or array [pu]   sending-end voltage magnitude
            delta_s_rad : float or array [rad]  sending-end voltage angle
            P_load_pu   : float or array [pu]   receiving-end active power demand
            Q_load_pu   : float or array [pu]   receiving-end reactive power demand
            length_km   : float (optional)      line length in km
        returns:
            V_r_pu          : [pu]   receiving-end voltage magnitude
            delta_r_rad     : [rad]  receiving-end voltage angle
            I_series_pu     : [pu]   series (line) current magnitude
            P_loss_pu       : [pu]   active power loss (I^2*R)
            Q_loss_pu       : [pu]   reactive power loss (I^2*X)
            P_s_pu          : [pu]   sending-end active power
            Q_s_pu          : [pu]   sending-end reactive power
            efficiency      : [—]    P_r / P_s
            voltage_drop_pu : [pu]   |V_s| - |V_r|
        """
        length_km = inputs.get("length_km", None)
        return self._model.compute(
            V_s_pu=inputs["V_s_pu"],
            delta_s_rad=inputs["delta_s_rad"],
            P_load_pu=inputs["P_load_pu"],
            Q_load_pu=inputs["Q_load_pu"],
            length_km=length_km,
        )

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Transmission Line (Overhead, Pi-Model)",
            "ec_id": "EC181",
            "fidelity": "F1a",
            "description": "Lumped pi-model: V_r from iterative load-flow; P_loss=I^2*R; Q_loss=I^2*X",
            "inputs": {
                "V_s_pu": {"unit": "pu", "range": [0.8, 1.2]},
                "delta_s_rad": {"unit": "rad", "range": [-0.524, 0.524]},
                "P_load_pu": {"unit": "pu", "range": [0.0, 2.0]},
                "Q_load_pu": {"unit": "pu", "range": [-1.0, 1.0]},
                "length_km": {"unit": "km", "range": [10.0, 1000.0], "optional": True},
            },
            "outputs": {
                "V_r_pu": {"unit": "pu"},
                "delta_r_rad": {"unit": "rad"},
                "I_series_pu": {"unit": "pu"},
                "P_loss_pu": {"unit": "pu"},
                "Q_loss_pu": {"unit": "pu"},
                "P_s_pu": {"unit": "pu"},
                "Q_s_pu": {"unit": "pu"},
                "efficiency": {"unit": "dimensionless"},
                "voltage_drop_pu": {"unit": "pu"},
            },
            "params": {
                "V_base": f"{u['V_base_kV']['value']} kV",
                "S_base": f"{u['S_base_MVA']['value']} MVA",
                "R_pu_per_km": u["R_pu_per_km"]["value"],
                "X_pu_per_km": u["X_pu_per_km"]["value"],
                "B_pu_per_km": u["B_pu_per_km"]["value"],
                "default_length_km": u["length_km"]["value"],
            },
            "source": "Glover, Sarma, Overbye (2012), Power Systems Analysis and Design, 5th ed.",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"V_s_pu": 1.05, "delta_s_rad": 0.1, "P_load_pu": 0.6, "Q_load_pu": 0.2})
    print(f"V_r={float(r['V_r_pu']):.4f} pu  "
          f"P_loss={float(r['P_loss_pu']):.4f} pu  "
          f"eta={float(r['efficiency'])*100:.2f}%  "
          f"dV={float(r['voltage_drop_pu']):.4f} pu")
