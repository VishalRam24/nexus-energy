"""EC043 -- Hybrid Supercapacitor -- F1b RC-Thermal -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import HybridSupercapacitorF1b


class ComponentModel:
    """Standardized interface for EC043 Hybrid Supercapacitor -- F1b RC-thermal model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = HybridSupercapacitorF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict hybrid supercapacitor operating point.

        Args:
            inputs: dict with keys:
                - v_cap (V):         Capacitor voltage (state variable)
                - current (A):       Terminal current (positive=discharge)
                - temperature (K):   Cell temperature (default: T_ref)

        Returns:
            dict with terminal_voltage_V, power_W, heat_W, esr_Ohm,
                      capacitance_F, soc, stored_energy_J, dvcap_dt_V_s
        """
        v_cap   = np.asarray(inputs["v_cap"], dtype=float)
        current = np.asarray(inputs["current"], dtype=float)
        temp    = np.asarray(inputs.get("temperature", self._model.T_ref), dtype=float)

        return {
            "terminal_voltage_V":  self._model.terminal_voltage(v_cap, current, temp),
            "power_W":             self._model.power(v_cap, current, temp),
            "heat_W":              self._model.heat_generation(current, temp),
            "esr_Ohm":             self._model.esr(temp),
            "capacitance_F":       self._model.capacitance(temp),
            "soc":                 self._model.soc(v_cap),
            "stored_energy_J":     self._model.stored_energy(v_cap, temp),
            "dvcap_dt_V_s":        self._model.vcap_derivative(v_cap, current, temp),
        }

    def get_info(self) -> dict:
        return {
            "name": "Hybrid Supercapacitor (Lithium-Ion Capacitor)",
            "ec_id": "EC043",
            "fidelity": "F1b",
            "description": (
                "RC thermal model: ESR(T) via Arrhenius, C(T) linear. "
                "Battery-type anode + EDLC cathode. V range 1.8-3.8 V."
            ),
            "inputs": {
                "v_cap":       {"unit": "V",             "range": [1.8, 3.8]},
                "current":     {"unit": "A",             "range": [-500.0, 500.0]},
                "temperature": {"unit": "K",             "range": [233.15, 333.15]},
            },
            "outputs": {
                "terminal_voltage_V":  {"unit": "V"},
                "power_W":             {"unit": "W"},
                "heat_W":              {"unit": "W"},
                "esr_Ohm":             {"unit": "Ohm"},
                "capacitance_F":       {"unit": "F"},
                "soc":                 {"unit": "dimensionless", "note": "Energy-based: (V^2-Vmin^2)/(Vmax^2-Vmin^2)"},
                "stored_energy_J":     {"unit": "J"},
                "dvcap_dt_V_s":        {"unit": "V/s"},
            },
            "source": "Zhang & Zhao (2009); Naoi (2012); Berrueta (2019)",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("EC043 Hybrid Supercapacitor -- F1b RC-Thermal")
    for T in [253.15, 273.15, 298.15, 318.15, 333.15]:
        r = model.predict({"v_cap": 3.0, "current": 50.0, "temperature": T})
        print(f"  T={T:.0f}K: V_term={float(r['terminal_voltage_V']):.3f}V, "
              f"ESR={float(r['esr_Ohm'])*1000:.3f}mOhm, "
              f"C={float(r['capacitance_F']):.1f}F")
