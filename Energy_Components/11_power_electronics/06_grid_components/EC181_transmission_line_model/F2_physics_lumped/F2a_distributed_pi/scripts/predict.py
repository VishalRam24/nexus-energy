"""
EC181 — Transmission Line Model — F2a Distributed-Parameter / Cascaded-Pi
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import TransmissionLineF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for EC181 F2a distributed-parameter transmission line."""

    component_id = "EC181"
    component_name = "Transmission Line Model"
    fidelity = "F2a — Distributed-Parameter / Cascaded-Pi Lumped Dynamic Model"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = TransmissionLineF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Steady-state transmission-line analysis (default) plus distributed quantities.

        inputs:
            V_s_pu      : sending-end voltage magnitude [pu]   (default 1.0)
            P_load_pu   : receiving-end active load [pu]       (default 1.0)
            Q_load_pu   : receiving-end reactive load [pu]     (default 0.0)
            length_km   : line length [km]                     (default from params)
            exact       : use exact hyperbolic ABCD (True) vs nominal-pi (False)

        Returns steady-state load-flow + ABCD + SIL + Ferranti diagnostics.
        """
        V_s = inputs.get("V_s_pu", 1.0)
        P = inputs.get("P_load_pu", 1.0)
        Q = inputs.get("Q_load_pu", 0.0)
        length = inputs.get("length_km", None)
        exact = inputs.get("exact", True)

        m = self._model
        res = m.solve_receiving(V_s, P, Q, length_km=length, exact=exact)
        A, B, Cc, D = m.abcd_exact(length) if exact else m.abcd_nominal_pi(length)
        ferr = m.ferranti_no_load(V_s, length_km=length, exact=exact)
        gamma, Z_c = m.gamma_zc()

        res.update({
            "ABCD": {"A": A, "B": B, "C": Cc, "D": D},
            "reciprocity_residual": abs(A * D - B * Cc - 1.0),
            "SIL_MW": m.sil_MW(),
            "surge_impedance_ohm": m.surge_impedance(),
            "Z_char_ohm": Z_c,
            "gamma_per_km": gamma,
            "ferranti": ferr,
        })
        return res

    def simulate(self, inputs: dict) -> dict:
        """Time-domain cascaded-pi dynamic simulation (delegates to model.simulate)."""
        return self._model.simulate(
            inputs.get("V_s_pu", 1.0),
            P_load_pu=inputs.get("P_load_pu", 1.0),
            R_load_pu=inputs.get("R_load_pu", None),
            L_load_pu=inputs.get("L_load_pu", None),
            n_sections=inputs.get("n_sections", 8),
            length_km=inputs.get("length_km", None),
            duration_s=inputs.get("duration_s", None),
            open_end=inputs.get("open_end", False),
        )

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "V_s_pu": {"unit": "pu", "range": [0.8, 1.2]},
                "P_load_pu": {"unit": "pu", "range": [0.0, 3.0]},
                "Q_load_pu": {"unit": "pu", "range": [-2.0, 2.0]},
                "length_km": {"unit": "km", "range": [10.0, 1000.0]},
                "n_sections": {"unit": "-", "range": [1, 50]},
                "exact": {"unit": "bool"},
            },
            "outputs": {
                "V_r_pu": "pu", "I_s_pu": "pu", "P_loss_pu": "pu",
                "efficiency": "-", "voltage_drop_pu": "pu",
                "ABCD": "dict (complex pu)", "SIL_MW": "MW",
                "ferranti": "dict", "(simulate) t/v_s/v_r/i_s/p_in/p_loss": "arrays",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    info = m.get_info()
    print(f"{info['component_id']} {info['component_name']} — {info['fidelity']}")
    r = m.predict({"V_s_pu": 1.0, "P_load_pu": 1.0, "Q_load_pu": 0.3, "length_km": 300.0})
    print(f"V_r = {r['V_r_pu']:.4f} pu | efficiency = {r['efficiency']*100:.2f}% "
          f"| P_loss = {r['P_loss_pu']*100:.3f} MW (on 100 MVA base)")
    print(f"SIL = {r['SIL_MW']:.1f} MW | surge Z = {r['surge_impedance_ohm']:.1f} ohm "
          f"| reciprocity |AD-BC-1| = {r['reciprocity_residual']:.2e}")
    f = r["ferranti"]
    print(f"No-load (Ferranti): V_r = {f['V_r_pu']:.4f} pu, rise x{f['rise_factor']:.4f}, "
          f"ferranti={f['ferranti']}")
    sim = m.simulate({"P_load_pu": 1.0, "n_sections": 6, "duration_s": 0.05})
    print(f"Dynamic sim: {len(sim['t'])} steps, success={sim['success']}, "
          f"final v_r={sim['v_r'][-1]:.4f} pu")
