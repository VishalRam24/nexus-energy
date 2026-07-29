"""
EC222 — Betavoltaic Cell — F2a Single-Diode I-V with Beta-Flux Photocurrent and Decay ODE
Standardised predict() / get_info() interface (mirrors EC001 template).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import BetavoltaicF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH, "r") as f:
        return json.load(f)


class ComponentModel:
    """Standardised wrapper for the EC222 betavoltaic F2a single-diode model."""

    component_id = "EC222"
    component_name = "Betavoltaic Cell"
    fidelity = "F2a — Single-Diode I-V with Beta-Flux Photocurrent and Decay ODE"
    version = "1.0.0"

    def __init__(self, params: dict = None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = BetavoltaicF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Run a betavoltaic life simulation (decay + single-diode MPP + thermal ODE).

        inputs:
            t_years    : float — end time of simulation [years]      (default 50.0)
            t0_years   : float — start time [years]                  (default 0.0)
            n_eval     : int   — number of output samples            (default 50)
            T_cell_K   : float — initial cell temperature [K]        (default ambient)
            with_iv    : bool  — solve single-diode MPP each sample  (default True)

        Returns dict of arrays over time (t_years, activity_Bq, fraction_remaining,
        P_beta_total_W, P_beta_absorbed_W, temperature_K, Isc_uA, Voc_V, FF,
        P_out_W, P_out_uW, eta) plus scalar snapshot fields at t0 for convenience.
        """
        t1 = inputs.get("t_years", 50.0)
        t0 = inputs.get("t0_years", 0.0)
        n_eval = int(inputs.get("n_eval", 50))
        T0 = inputs.get("T_cell_K", None)
        with_iv = inputs.get("with_iv", True)

        result = self._model.simulate((t0, t1), n_eval=n_eval, T0_K=T0, with_iv=with_iv)

        # Convenience scalar snapshot at the start time (full I-V curve)
        iv0 = self._model.iv_curve(t0, T0 if T0 is not None else self._model.T_amb)
        result["snapshot_t0"] = {
            "Isc_uA": iv0["Isc_A"] * 1e6,
            "Voc_V": iv0["Voc_V"],
            "FF": iv0["FF"],
            "P_mpp_uW": iv0["P_mpp_W"] * 1e6,
            "V_mpp_V": iv0["V_mpp_V"],
            "ehp_per_beta": self._model.ehp_per_beta(),
            "E_pair_eV": self._model.pair_creation_energy_eV(),
        }
        return result

    def iv_curve(self, t_years=0.0, T_cell_K=None):
        """Expose a single I-V / P-V sweep at a given time (for plotting)."""
        return self._model.iv_curve(t_years, T_cell_K)

    def get_info(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "fidelity": self.fidelity,
            "version": self.version,
            "inputs": {
                "t_years": {"unit": "years", "range": [0.0, 500.0]},
                "t0_years": {"unit": "years", "range": [0.0, 500.0]},
                "n_eval": {"unit": "-", "range": [2, 5000]},
                "T_cell_K": {"unit": "K", "range": [200.0, 500.0]},
                "with_iv": {"unit": "bool"},
            },
            "outputs": {
                "t_years": "years",
                "activity_Bq": "Bq",
                "fraction_remaining": "-",
                "P_beta_total_W": "W",
                "P_beta_absorbed_W": "W",
                "temperature_K": "K",
                "Isc_uA": "uA",
                "Voc_V": "V",
                "FF": "-",
                "P_out_W": "W",
                "P_out_uW": "uW",
                "eta": "-",
            },
            "source": self._raw.get("source", ""),
        }


if __name__ == "__main__":
    m = ComponentModel()
    print(m.get_info())
    r = m.predict({"t_years": 50.0, "n_eval": 11})
    s = r["snapshot_t0"]
    print(f"\nt=0 snapshot: Isc={s['Isc_uA']:.4f} uA, Voc={s['Voc_V']:.4f} V, "
          f"FF={s['FF']:.3f}, P_mpp={s['P_mpp_uW']:.4f} uW")
    print(f"  EHP per beta = {s['ehp_per_beta']:.3f}, E_pair = {s['E_pair_eV']:.2f} eV")
    print(f"P_out: t=0 -> {r['P_out_uW'][0]:.4f} uW, "
          f"t=50yr -> {r['P_out_uW'][-1]:.4f} uW "
          f"(fraction remaining {r['fraction_remaining'][-1]:.3f})")
    print(f"eta at t=0: {r['eta'][0]*100:.2f} %, T = {r['temperature_K'][0]:.4f} K")
