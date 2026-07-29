"""EC079 -- Molten Salt TES -- F1b Stratified 10-Node -- Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import MoltenSaltTESF1b


class ComponentModel:
    """Standardized interface for EC079 Molten Salt TES -- F1b stratified model."""

    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = MoltenSaltTESF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Run a single operating step of the stratified TES model.

        Args:
            inputs: {
                "T_charge_degC":     float, inlet temp during charging [degC]
                "T_discharge_degC":  float, inlet temp during discharging [degC]
                "flow_rate_kg_s":    float, mass flow rate [kg/s]
                "mode":              str, 'charge' | 'discharge' | 'idle'
                "T_ambient_degC":    float, ambient temp [degC] (default 25)
                "duration_s":        float, step duration [s] (default 3600)
                "T_nodes_init":      list of 10 floats [degC] (optional)
            }

        Returns:
            dict with:
                T_nodes            : list of 10 node temperatures [degC]
                T_outlet_degC      : outlet temperature [degC]
                stored_energy_kwh  : total stored energy [kWh]
                thermal_efficiency : efficiency [-]
                freeze_warning     : bool
        """
        return self._model.predict(
            T_charge_degC=float(inputs.get("T_charge_degC", 565.0)),
            T_discharge_degC=float(inputs.get("T_discharge_degC", 290.0)),
            flow_rate_kg_s=float(inputs.get("flow_rate_kg_s", 0.0)),
            mode=str(inputs.get("mode", "idle")),
            T_ambient_degC=float(inputs.get("T_ambient_degC", 25.0)),
            duration_s=float(inputs.get("duration_s", 3600.0)),
            T_nodes_init=inputs.get("T_nodes_init", None),
        )

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Molten Salt Thermal Energy Storage",
            "ec_id": "EC079",
            "fidelity": "F1b",
            "model": "10-Node Stratified with T-Dependent Properties",
            "description": (
                "10-node vertical stratification model for solar salt TES. "
                "Temperature-dependent properties: rho(T)=2090-0.636*T, cp(T)=1443+0.172*T. "
                f"Tank: {m.volume} m3, {m.height} m tall. "
                f"Operating range: {m.T_cold_design}-{m.T_hot_design} degC. "
                f"Freeze limit: {m.T_freeze} degC."
            ),
            "inputs": {
                "T_charge_degC":    {"unit": "degC", "range": [290.0, 600.0]},
                "T_discharge_degC": {"unit": "degC", "range": [220.0, 565.0]},
                "flow_rate_kg_s":   {"unit": "kg/s", "range": [0.0, 2000.0]},
                "mode":             {"values": ["charge", "discharge", "idle"]},
                "T_ambient_degC":   {"unit": "degC", "range": [-20.0, 60.0], "default": 25.0},
                "duration_s":       {"unit": "s", "range": [0.0, 86400.0], "default": 3600.0},
                "T_nodes_init":     {"unit": "degC", "shape": [10], "optional": True},
            },
            "outputs": {
                "T_nodes":            {"unit": "degC", "shape": [10]},
                "T_outlet_degC":      {"unit": "degC"},
                "stored_energy_kwh":  {"unit": "kWh"},
                "thermal_efficiency": {"unit": "dimensionless"},
                "freeze_warning":     {"unit": "bool"},
            },
            "source": "Herrmann et al. (2004); Pacheco et al. (2002); Zaversky et al. (2013)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    info = model.get_info()
    print(f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}")

    # Charging example
    T_init = np.linspace(290, 350, 10).tolist()
    r = model.predict({
        "T_charge_degC": 565.0,
        "T_discharge_degC": 290.0,
        "flow_rate_kg_s": 500.0,
        "mode": "charge",
        "T_ambient_degC": 25.0,
        "duration_s": 3600.0,
        "T_nodes_init": T_init,
    })
    print(f"\nCharging for 1 hour at 500 kg/s:")
    print(f"  T_nodes: {[f'{t:.1f}' for t in r['T_nodes']]}")
    print(f"  T_outlet: {r['T_outlet_degC']:.1f} degC")
    print(f"  Stored energy: {r['stored_energy_kwh']:.1f} kWh")
    print(f"  Freeze warning: {r['freeze_warning']}")
