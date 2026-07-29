"""
EC173 -- Distribution Transformer -- F2a Equivalent-Circuit + Thermal ODE
Standardised predict() / get_info() interface (mirrors EC001 F2a).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import DistributionTransformerF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC173 distribution transformer F2a model."""

    component_id = "EC173"
    component_name = "Distribution Transformer"
    fidelity = "F2a -- Equivalent Circuit + Lumped Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = DistributionTransformerF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a transformer operating-point + transient thermal simulation.

        inputs:
            load_fraction      : float (per-unit load K, default 0.5).
            voltage_pu         : float (default 1.0)
            power_factor       : float (default 0.9 lagging)
            ambient_temperature: float degC (default 20.0)
            dt                 : float s (default 60.0)
            duration_s         : float s (default 14400.0 = 4 h)
            daily              : bool -- if True use 24h residential profile
                                 (ignores scalar load_fraction), default False.
        Returns operating-point scalars + thermal time-series arrays.
        """
        m = self._model
        K = inputs.get("load_fraction", 0.5)
        v = inputs.get("voltage_pu", 1.0)
        pf = inputs.get("power_factor", 0.9)
        T_amb = inputs.get("ambient_temperature", 20.0)
        dt = inputs.get("dt", 60.0)
        dur = inputs.get("duration_s", 14400.0)
        daily = inputs.get("daily", False)

        if daily:
            prof, hourly = m.residential_daily_profile()
            load_input = prof
            dur = max(dur, 86400.0)
            K_point = float(hourly.mean())  # average load factor
        else:
            load_input = K
            K_point = float(K)

        # Steady-state electrical operating point (at average/scalar load).
        eta = float(m.efficiency(K_point, v, T_amb + 65.0, pf))
        losses = {
            "p_core_w": float(m.core_loss(v)),
            "p_copper_w": float(m.copper_loss(K_point, T_amb + 65.0)),
            "p_total_w": float(m.total_loss(K_point, v, T_amb + 65.0)),
        }
        vr = m.voltage_regulation(K_point, pf)

        sim = m.simulate_thermal(load_input, T_amb, dt=dt, duration=dur)

        return {
            "operating_point": {
                "load_fraction": K_point,
                "efficiency": eta,
                "optimal_load_fraction": m.optimal_load_fraction(),
                "voltage_regulation_pu": vr["vr_exact_pu"],
                "losses_w": losses,
            },
            "equivalent_circuit": m.equivalent_circuit(),
            # Thermal transient arrays
            "t": sim["t"],
            "T_hot_spot": sim["T_hot_spot"],
            "T_top_oil": sim["T_top_oil"],
            "load": sim["load"],
            "p_total": sim["p_total"],
            "T_hot_spot_final": float(sim["T_hot_spot"][-1]),
        }

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "load_fraction": {"unit": "pu", "range": [0.0, 2.0]},
                "voltage_pu": {"unit": "pu", "range": [0.9, 1.1]},
                "power_factor": {"unit": "-", "range": [0.5, 1.0]},
                "ambient_temperature": {"unit": "degC", "range": [-25, 50]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
                "daily": {"unit": "bool"},
            },
            "outputs": {
                "operating_point": "dict (efficiency, VR, losses)",
                "equivalent_circuit": "dict (R_eq, X_eq, R_c, X_m) pu",
                "t": "s",
                "T_hot_spot": "degC",
                "T_top_oil": "degC",
                "load": "pu",
                "p_total": "W",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"load_fraction": 0.5, "power_factor": 0.9, "duration_s": 14400.0})
    op = r["operating_point"]
    print(f"eta={op['efficiency']*100:.2f}%  PLR_opt={op['optimal_load_fraction']:.3f}  "
          f"VR={op['voltage_regulation_pu']*100:.2f}%  "
          f"T_hotspot_final={r['T_hot_spot_final']:.1f} degC")
