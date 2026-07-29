"""
EC156 -- Geothermal Heat Pump (GHP / Ground-Source) -- F2a
Vapor-compression cycle + lumped borehole/condenser thermal ODE.
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import GHP_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for GHP F2a vapor-compression + borehole ODE model."""

    component_id = "EC156"
    component_name = "Geothermal Heat Pump (GHP / Ground-Source)"
    fidelity = "F2a -- Vapor-Compression Cycle + Lumped Borehole/Condenser ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = GHP_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a transient GSHP simulation.

        inputs:
            T_supply_c  : float  building heating supply temperature (degC), default 45
            Q_demand_kW : float  building heating demand (kW), default 8
            n_comp      : float  compressor speed (rev/s), default rated
            dt          : float  output step (s), default 600
            duration_s  : float  total simulated time (s), default 86400 (1 day)
            T_ground_undisturbed : float (degC) optional override
        """
        T_supply = inputs.get("T_supply_c", 45.0)
        Q_dem = inputs.get("Q_demand_kW", 8.0)
        n_comp = inputs.get("n_comp", None)
        dt = inputs.get("dt", 600.0)
        dur = inputs.get("duration_s", 86400.0)
        Tg0 = inputs.get("T_ground_undisturbed", None)

        return self._model.simulate(
            T_supply_c=T_supply, Q_demand_kW=Q_dem, n_comp=n_comp,
            dt=dt, duration_s=dur, T_ground0=Tg0,
        )

    def operating_point(self, T_loop_c, T_supply_c, n_comp=None) -> dict:
        """Steady cycle metrics (COP, duties) at a fixed loop/supply temperature."""
        return self._model.cycle(T_loop_c, T_supply_c, n_comp)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_supply_c": {"unit": "degC", "range": [25, 60]},
                "Q_demand_kW": {"unit": "kW", "range": [0, 12]},
                "n_comp": {"unit": "rev/s", "range": [20, 60]},
                "dt": {"unit": "s", "range": [1, 600]},
                "duration_s": {"unit": "s", "range": [60, 864000]},
            },
            "outputs": {
                "t": "s",
                "T_loop": "degC (ground-loop fluid)",
                "T_ground": "degC (lumped ground node)",
                "T_cond_node": "degC (condenser/building-water node)",
                "T_evap": "degC (refrigerant evaporation)",
                "COP": "- (heating COP)",
                "COP_carnot": "- (Carnot ceiling)",
                "Q_cond_kW": "kW (heat delivered)",
                "Q_evap_kW": "kW (heat drawn from ground)",
                "W_elec_kW": "kW (electrical input)",
            },
            "refrigerant": self._raw["unit"].get("refrigerant", "R-410A"),
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    op = m.operating_point(8.0, 45.0)
    print(f"Rated point: COP={op['COP']:.2f} (Carnot {op['COP_carnot']:.2f}), "
          f"Q_cond={op['Q_cond']/1e3:.2f} kW, W_elec={op['W_elec']/1e3:.2f} kW")
    r = m.predict({"T_supply_c": 45.0, "Q_demand_kW": 8.0, "dt": 600.0,
                   "duration_s": 5 * 86400})
    print(f"5-day sim: T_loop {r['T_loop'][0]:.2f} -> {r['T_loop'][-1]:.2f} degC, "
          f"COP {r['COP'][0]:.2f} -> {r['COP'][-1]:.2f}, success={r['success']}")
