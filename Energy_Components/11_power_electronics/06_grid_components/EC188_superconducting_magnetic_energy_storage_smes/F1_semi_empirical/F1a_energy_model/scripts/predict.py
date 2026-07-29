"""EC188 — SMES — F1a Energy Model — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import SMESModel


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = SMESModel(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            SOC            : float or array [—]   state of charge (energy-based, 0–1)
            P_request_MW   : float or array [MW]  power request (+ always)
            mode           : str  "charge" or "discharge" (default "discharge")
            dt_s           : float (optional)     time step [s] for SOC update (default 1.0)
        returns:
            P_delivered_MW       : [MW]  power delivered to/from converter
            P_grid_MW            : [MW]  net power seen by grid (- = consuming)
            P_cryo_MW            : [MW]  cryogenic refrigeration power (constant)
            P_total_parasitic_MW : [MW]  total parasitic losses
            SOC_new              : [—]   SOC after dt_s
            E_stored_MJ          : [MJ]  stored energy after dt_s
            eta_instantaneous    : [—]   instantaneous efficiency
            dE_MJ                : [MJ]  energy change in dt_s
            mode                 : str
        """
        return self._model.compute(
            SOC=inputs["SOC"],
            P_request_MW=inputs["P_request_MW"],
            mode=inputs.get("mode", "discharge"),
            dt_s=inputs.get("dt_s", 1.0),
        )

    def energy_from_current(self, inputs: dict) -> dict:
        """
        inputs:
            I_A : float or array [A]  coil current
        returns:
            E_MJ  : [MJ]  stored energy
            SOC   : [—]   state of charge
            I_coil_A: [A] clamped coil current
        """
        return self._model.energy_from_current(I_A=inputs["I_A"])

    def get_info(self) -> dict:
        u = self.params["unit"]
        return {
            "name": "Superconducting Magnetic Energy Storage (SMES)",
            "ec_id": "EC188",
            "fidelity": "F1a",
            "description": "E=0.5*L*I^2; SOC=E/E_max; P_cryo continuous; eta=eta_conv^2 approx",
            "inputs": {
                "SOC": {"unit": "dimensionless", "range": [0.0, 1.0]},
                "P_request_MW": {"unit": "MW", "range": [0.0, 10.0]},
                "mode": {"unit": "enum", "values": ["charge", "discharge"]},
                "dt_s": {"unit": "s", "optional": True},
            },
            "outputs": {
                "P_delivered_MW": {"unit": "MW"},
                "P_grid_MW": {"unit": "MW"},
                "P_cryo_MW": {"unit": "MW"},
                "SOC_new": {"unit": "dimensionless"},
                "E_stored_MJ": {"unit": "MJ"},
                "eta_instantaneous": {"unit": "dimensionless"},
                "dE_MJ": {"unit": "MJ"},
            },
            "params": {
                "L": f"{u['L_H']['value']} H",
                "I_max": f"{u['I_max_A']['value']} A",
                "E_max": f"{self._model.E_max_MJ:.1f} MJ",
                "P_rated": f"{u['P_rated_MW']['value']} MW",
                "P_cryo": f"{u['P_cryo_MW']['value']} MW",
                "eta_converter": u["eta_converter"]["value"],
                "T_operating": f"{u['T_operating_K']['value']} K",
            },
            "source": "Buckles & Hassenzahl (2000), IEEE Power Eng. Rev.; Kalsi (2011), Wiley",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"E_max = {model._model.E_max_MJ:.2f} MJ")
    for soc in [1.0, 0.75, 0.5, 0.25, 0.1]:
        r = model.predict({"SOC": soc, "P_request_MW": 8.0, "mode": "discharge"})
        print(f"SOC={soc:.2f}  P_grid={float(r['P_grid_MW']):.2f} MW  "
              f"eta={float(r['eta_instantaneous'])*100:.2f}%  E={float(r['E_stored_MJ']):.2f} MJ")
