"""EC154 — Enhanced Geothermal System (EGS) — F1a Exergy Model — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import EGSF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = EGSF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict EGS plant performance.

        Parameters
        ----------
        inputs : dict
            T_geothermal  : float or array  (degC, 150–350) — rock temperature at depth
            T_rejection   : float or array  (degC, 10–40)   — surface cooling rejection T
            flow_rate_kgs : float or array  (kg/s, 5–200)   — circulation flow rate

        Returns
        -------
        dict
            power_net_kw     : net electrical output after parasitic deduction (kW)
            power_gross_kw   : gross electrical output (kW)
            parasitic_kw     : pump parasitic power (kW)
            efficiency_net   : net plant efficiency (-)
            efficiency_gross : gross cycle efficiency (-)
            heat_input_kw    : thermal energy extracted from rock (kW)
            T_reinjection_c  : reinjection temperature (degC)
        """
        T_geo  = np.asarray(inputs["T_geothermal"],  dtype=float)
        T_rej  = np.asarray(inputs["T_rejection"],   dtype=float)
        m_dot  = np.asarray(inputs["flow_rate_kgs"], dtype=float)

        P_net   = self._model.net_power(T_geo, T_rej, m_dot)
        P_gross = self._model.gross_power(T_geo, T_rej, m_dot)
        P_par   = self._model.parasitic_power(T_geo, T_rej, m_dot)
        eta_net = self._model.net_efficiency(T_geo, T_rej)
        eta_gro = self._model.gross_efficiency(T_geo, T_rej)
        Q_in    = self._model.heat_input(T_geo, T_rej, m_dot)
        T_reinj = self._model.reinjection_temperature(T_rej)

        return {
            "power_net_kw": P_net,
            "power_gross_kw": P_gross,
            "parasitic_kw": P_par,
            "efficiency_net": eta_net,
            "efficiency_gross": eta_gro,
            "heat_input_kw": Q_in,
            "T_reinjection_c": T_reinj,
        }

    def get_info(self) -> dict:
        return {
            "name": "Enhanced Geothermal System (EGS)",
            "ec_id": "EC154",
            "fidelity": "F1a",
            "description": (
                "Exergy model: eta_net = eta_util * eta_Carnot * (1 - pump_parasitic_frac); "
                "hot dry rock + hydraulic fracturing; binary/ORC conversion"
            ),
            "inputs": {
                "T_geothermal":  {"unit": "degC",  "range": [150.0, 350.0]},
                "T_rejection":   {"unit": "degC",  "range": [10.0,  40.0]},
                "flow_rate_kgs": {"unit": "kg/s",  "range": [5.0,   200.0]},
            },
            "outputs": {
                "power_net_kw":     {"unit": "kW"},
                "power_gross_kw":   {"unit": "kW"},
                "parasitic_kw":     {"unit": "kW"},
                "efficiency_net":   {"unit": "dimensionless"},
                "efficiency_gross": {"unit": "dimensionless"},
                "heat_input_kw":    {"unit": "kW"},
                "T_reinjection_c":  {"unit": "degC"},
            },
            "source": "Tester et al. (2006) MIT/DOE; DiPippo (2015) Ch. 16",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"T_geothermal": 200.0, "T_rejection": 25.0, "flow_rate_kgs": 50.0})
    print(f"T_geo=200°C, T_rej=25°C, m_dot=50 kg/s:")
    print(f"  Power (net)   = {float(r['power_net_kw']):.1f} kW")
    print(f"  Power (gross) = {float(r['power_gross_kw']):.1f} kW")
    print(f"  Parasitic     = {float(r['parasitic_kw']):.1f} kW")
    print(f"  η_net         = {float(r['efficiency_net']):.4f} ({float(r['efficiency_net'])*100:.2f}%)")
    print(f"  Heat input    = {float(r['heat_input_kw']):.1f} kW")
    print(f"  T_reinject    = {float(r['T_reinjection_c']):.1f} °C")
    info = model.get_info()
    print(f"  EC ID: {info['ec_id']}, Fidelity: {info['fidelity']}")
