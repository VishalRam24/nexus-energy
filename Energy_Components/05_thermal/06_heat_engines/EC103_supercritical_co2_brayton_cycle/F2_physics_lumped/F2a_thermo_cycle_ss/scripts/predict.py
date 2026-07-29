"""
EC103 -- Supercritical CO2 Brayton Cycle -- F2a Physics-Lumped
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import SCO2BraytonF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the sCO2 Brayton F2a physics-lumped model."""

    component_id = "EC103"
    component_name = "Supercritical CO2 Brayton Cycle"
    fidelity = "F2a -- Physics-Lumped Recompression Brayton Cycle with Thermal ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = SCO2BraytonF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Solve the steady-state cycle and (optionally) the transient warm-up.

        inputs:
            T_turb_in_K  : float  (turbine inlet temperature, default design)
            T_comp_in_K  : float  (main compressor inlet, near-critical)
            P_high_Pa    : float  (high-side pressure)
            P_low_Pa     : float  (low-side pressure)
            recompression_fraction : float (split fraction)
            transient    : bool   (if True, also run the lumped thermal ODE)
            T_metal0_K   : float  (initial hot-section temperature for transient)
            dt           : float  (s, transient step)
            duration_s   : float  (s, transient horizon)
        """
        T4 = inputs.get("T_turb_in_K", None)
        T1 = inputs.get("T_comp_in_K", None)
        Ph = inputs.get("P_high_Pa", None)
        Pl = inputs.get("P_low_Pa", None)
        f = inputs.get("recompression_fraction", None)

        cyc = self._model.cycle(T_turb_in=T4, T_comp_in=T1, P_high=Ph,
                                P_low=Pl, f_rc=f)

        out = {
            "eta_thermal": cyc["eta_thermal"],
            "eta_carnot": cyc["eta_carnot"],
            "w_net_J_per_kg": cyc["w_net"],
            "w_turbine_J_per_kg": cyc["w_turb"],
            "w_compressor_J_per_kg": cyc["w_comp"],
            "back_work_ratio": cyc["back_work_ratio"],
            "P_net_W": cyc["P_net_W"],
            "Q_in_W": cyc["Q_in_W"],
            "Q_rej_W": cyc["Q_rej_W"],
            "states_K": cyc["states"],
            "Z_comp_in": cyc["Z_comp_in"],
            "density_comp_in_kg_m3": cyc["density_comp_in"],
        }

        if inputs.get("transient", False):
            sim = self._model.simulate(
                T_metal0=inputs.get("T_metal0_K", None),
                dt=inputs.get("dt", 1.0),
                duration_s=inputs.get("duration_s", 600.0),
                T_turb_in=T4, T_comp_in=T1, P_high=Ph, P_low=Pl, f_rc=f,
            )
            out["transient"] = {
                "t": sim["t"],
                "T_turbine_inlet": sim["T_turbine_inlet"],
                "efficiency": sim["efficiency"],
                "P_net_W": sim["P_net_W"],
            }
        return out

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "T_turb_in_K": {"unit": "K", "range": [773.15, 1073.15]},
                "T_comp_in_K": {"unit": "K", "range": [305.15, 333.15]},
                "P_high_Pa": {"unit": "Pa", "range": [18e6, 30e6]},
                "P_low_Pa": {"unit": "Pa", "range": [7.5e6, 12e6]},
                "recompression_fraction": {"unit": "-", "range": [0.0, 0.6]},
                "transient": {"unit": "bool"},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
            },
            "outputs": {
                "eta_thermal": "-",
                "eta_carnot": "-",
                "w_net_J_per_kg": "J/kg",
                "back_work_ratio": "-",
                "P_net_W": "W",
                "Q_in_W": "W",
                "Q_rej_W": "W",
                "states_K": "dict of station temperatures (K)",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"transient": True, "duration_s": 300.0, "dt": 5.0})
    print(f"\neta_thermal = {r['eta_thermal']:.4f}  (Carnot = {r['eta_carnot']:.4f})")
    print(f"back-work ratio = {r['back_work_ratio']:.4f}  (low => sCO2 advantage)")
    print(f"P_net = {r['P_net_W']/1e6:.2f} MW,  Q_in = {r['Q_in_W']/1e6:.2f} MW")
    print(f"Compressor-inlet Z = {r['Z_comp_in']:.3f}, rho = {r['density_comp_in_kg_m3']:.1f} kg/m3")
    print(f"Final turbine-inlet T (transient) = {r['transient']['T_turbine_inlet'][-1]:.1f} K")
