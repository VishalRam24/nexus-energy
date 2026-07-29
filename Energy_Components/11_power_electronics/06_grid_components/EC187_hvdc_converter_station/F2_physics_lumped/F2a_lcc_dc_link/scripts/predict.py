"""
EC187 -- HVDC Converter Station -- F2a Physics-Lumped (LCC + DC-link ODE)
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import HVDC_LCC_F2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC187 HVDC LCC F2a physics-lumped model."""

    component_id = "EC187"
    component_name = "HVDC Converter Station (LCC, point-to-point link)"
    fidelity = "F2a -- LCC 12-pulse converters + DC-link current ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = HVDC_LCC_F2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a DC-link current transient for a commanded power / firing angle.

        inputs:
            P_order_MW   : float  -- commanded DC power; sets rectifier alpha
                           (ignored if alpha_deg supplied directly).
            alpha_deg    : float  -- rectifier firing angle [deg] (optional override).
            gamma_deg    : float  -- inverter extinction angle [deg] (default param).
            Id0_kA       : float  -- initial DC current [kA] (default 0).
            dt           : float  -- output time step [s] (default 1e-3).
            duration_s   : float  -- transient duration [s] (default 0.5).
            T_line_degC  : float  -- DC line temperature [degC] (default 20).

        Returns dict of time-series (SI / engineering units) plus the
        steady-state operating point reached.
        """
        m = self._model

        gamma_deg = inputs.get("gamma_deg", None)
        gamma = np.deg2rad(gamma_deg) if gamma_deg is not None else None

        T_line = inputs.get("T_line_degC", 20.0)

        if "alpha_deg" in inputs and inputs["alpha_deg"] is not None:
            alpha = np.deg2rad(inputs["alpha_deg"])
        else:
            P_order_W = inputs.get("P_order_MW", 0.5 * m.P_rated / 1e6) * 1e6
            alpha = m.alpha_for_power(P_order_W, gamma, T_line)

        Id0 = inputs.get("Id0_kA", 0.0) * 1e3
        dt = inputs.get("dt", 1e-3)
        dur = inputs.get("duration_s", 0.5)

        sim = m.simulate(alpha, gamma, Id0, dt, dur, T_line)

        # Convert to engineering units for the standardised output.
        out = {
            "t": sim["t"],
            "Id_kA": sim["Id_A"] / 1e3,
            "Vd_rect_kV": sim["Vd_rect_V"] / 1e3,
            "Vd_inv_kV": sim["Vd_inv_V"] / 1e3,
            "P_dc_rect_MW": sim["P_dc_rect_W"] / 1e6,
            "P_dc_inv_MW": sim["P_dc_inv_W"] / 1e6,
            "P_line_loss_MW": sim["P_line_loss_W"] / 1e6,
            "efficiency": sim["efficiency"],
            "Q_rect_MVAR": sim["Q_rect_VAR"] / 1e6,
            "Q_inv_MVAR": sim["Q_inv_VAR"] / 1e6,
            "alpha_deg": float(np.rad2deg(alpha)),
            "gamma_deg": float(np.rad2deg(gamma if gamma is not None else m.gamma_inv)),
        }

        # Steady-state operating point summary.
        Id_ss = m.steady_state_current(alpha, gamma, T_line)
        pb = m.power_balance(Id_ss, alpha, gamma, T_line)
        out["steady_state"] = {
            "Id_kA": Id_ss / 1e3,
            "Vd_rect_kV": pb["Vd_rect_V"] / 1e3,
            "Vd_inv_kV": pb["Vd_inv_V"] / 1e3,
            "P_transfer_MW": pb["P_dc_rect_W"] / 1e6,
            "P_ac_in_MW": pb["P_ac_in_W"] / 1e6,
            "P_ac_out_MW": pb["P_ac_out_W"] / 1e6,
            "efficiency": pb["efficiency"],
            "Q_rect_MVAR": pb["Q_rect_VAR"] / 1e6,
            "Q_inv_MVAR": pb["Q_inv_VAR"] / 1e6,
            "pf_rect": pb["pf_rect"],
            "pf_inv": pb["pf_inv"],
        }
        return out

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "P_order_MW": {"unit": "MW", "range": [0, 1000]},
                "alpha_deg": {"unit": "deg", "range": [5, 90]},
                "gamma_deg": {"unit": "deg", "range": [10, 40]},
                "Id0_kA": {"unit": "kA", "range": [0, 2.4]},
                "dt": {"unit": "s"},
                "duration_s": {"unit": "s"},
                "T_line_degC": {"unit": "degC", "range": [-20, 80]},
            },
            "outputs": {
                "t": "s",
                "Id_kA": "kA",
                "Vd_rect_kV": "kV",
                "Vd_inv_kV": "kV",
                "P_dc_rect_MW": "MW",
                "P_dc_inv_MW": "MW",
                "P_line_loss_MW": "MW",
                "efficiency": "-",
                "Q_rect_MVAR": "MVAR",
                "Q_inv_MVAR": "MVAR",
                "steady_state": "dict of operating-point values",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"P_order_MW": 1000.0, "duration_s": 0.5, "dt": 5e-3})
    ss = r["steady_state"]
    print(
        f"\nRated transfer: Id={ss['Id_kA']:.3f} kA, "
        f"Vd_rect={ss['Vd_rect_kV']:.1f} kV, P={ss['P_transfer_MW']:.1f} MW, "
        f"eta={ss['efficiency']:.4f}, "
        f"Q_rect={ss['Q_rect_MVAR']:.0f} MVAR (Q/P={ss['Q_rect_MVAR']/ss['P_transfer_MW']:.2f})"
    )
    print(f"Final dynamic Id = {r['Id_kA'][-1]:.3f} kA (firing alpha={r['alpha_deg']:.1f} deg)")
