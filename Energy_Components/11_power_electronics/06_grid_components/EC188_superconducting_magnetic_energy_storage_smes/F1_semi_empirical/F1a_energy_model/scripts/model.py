"""
EC188 — SMES — F1a Energy Model

Stored energy:
    E = 0.5 * L * I^2    [J]
    E_MJ = E / 1e6

State of charge (energy-based):
    SOC = E / E_max = (I / I_max)^2

Charge/discharge:
    P_charge_net   = P_AC * eta_conv    [into DC side]
    P_discharge_net = P_DC * eta_conv  [out to AC side]

Round-trip efficiency:
    eta_RT = eta_conv^2 - P_cryo * t_cycle / E_max    (approx)
    At F1a: eta_RT ≈ eta_conv^2 * (1 - P_cryo/P_discharge)

Cryogenic losses are continuous (always drawn regardless of charge/discharge state).

Reference:
    Buckles, W. & Hassenzahl, W.V. (2000). Superconducting magnetic energy storage.
    IEEE Power Eng. Rev. 20(5):16-20.
    Kalsi, S.S. (2011). Applications of HTS Superconductors. Wiley.
"""

import numpy as np


class SMESModel:
    """SMES F1a energy storage model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.L = u["L_H"]["value"]                    # H
        self.I_max = u["I_max_A"]["value"]             # A
        self.I_min = u["I_min_A"]["value"]             # A
        self.P_rated = u["P_rated_MW"]["value"]        # MW
        self.P_cryo = u["P_cryo_MW"]["value"]          # MW (continuous)
        self.eta_conv = u["eta_converter"]["value"]    # dimensionless

        self.E_max = 0.5 * self.L * self.I_max ** 2   # J
        self.E_max_MJ = self.E_max / 1e6

    def energy_from_current(self, I_A: float) -> dict:
        """Compute stored energy and SOC from coil current."""
        I = np.asarray(I_A, dtype=float)
        I_clamped = np.clip(I, self.I_min, self.I_max)
        E = 0.5 * self.L * I_clamped ** 2
        E_MJ = E / 1e6
        SOC = E / (self.E_max + 1e-30)
        return {"E_MJ": E_MJ, "SOC": SOC, "I_coil_A": I_clamped}

    def compute(self, SOC: float, P_request_MW: float,
                mode: str = "discharge", dt_s: float = 1.0) -> dict:
        """
        Parameters
        ----------
        SOC           : State of charge [0–1], energy-based
        P_request_MW  : Requested power [MW] (positive always)
        mode          : "charge" or "discharge"
        dt_s          : Time step [s] for energy update

        Returns
        -------
        dict with P_delivered_MW, P_grid_MW, P_cryo_MW, P_total_parasitic_MW,
        SOC_new, E_stored_MJ, eta_instantaneous, dE_MJ
        """
        SOC = np.asarray(np.clip(SOC, 0.0, 1.0), dtype=float)
        P_req = np.asarray(np.clip(P_request_MW, 0.0, self.P_rated), dtype=float)

        if mode == "discharge":
            # Cannot discharge more than available energy
            E_avail_MJ = SOC * self.E_max_MJ
            P_max_from_SOC = E_avail_MJ / (dt_s / 1e6 + 1e-30) if dt_s > 0 else self.P_rated
            P_dc = np.minimum(P_req, self.P_rated)
            P_ac = P_dc * self.eta_conv  # AC output after converter loss
            # Total power to grid (minus cryogenic parasitics)
            P_grid = P_ac - self.P_cryo
            P_grid = np.maximum(P_grid, 0.0)
            dE = -P_dc * dt_s / 1e6   # MJ (decrease)
        else:  # charge
            P_ac = P_req  # AC input from grid
            P_dc = P_ac * self.eta_conv  # into coil after converter loss
            P_grid = -P_ac - self.P_cryo  # grid sees charging + cryo demand
            dE = P_dc * dt_s / 1e6   # MJ (increase)

        SOC_new = np.clip(SOC + dE / (self.E_max_MJ + 1e-30), 0.0, 1.0)
        E_stored_MJ = SOC_new * self.E_max_MJ

        # Instantaneous efficiency
        if mode == "discharge":
            safe_Pdc = np.where(P_dc > 0, P_dc, 1e-12) if np.ndim(P_dc) > 0 else (P_dc if P_dc > 0 else 1e-12)
            eta = np.where(P_dc > 0, (P_ac - self.P_cryo) / safe_Pdc, 0.0) if np.ndim(P_dc) > 0 else ((P_ac - self.P_cryo) / safe_Pdc if P_dc > 0 else 0.0)
            eta = np.maximum(eta, 0.0)
        else:
            safe_Pac = np.where(P_ac > 0, P_ac + self.P_cryo, 1e-12) if np.ndim(P_ac) > 0 else ((P_ac + self.P_cryo) if P_ac > 0 else 1e-12)
            eta = np.where(P_ac > 0, (P_dc) / safe_Pac, 0.0) if np.ndim(P_ac) > 0 else (P_dc / safe_Pac if P_ac > 0 else 0.0)

        return {
            "P_delivered_MW": P_ac if mode == "discharge" else P_dc,
            "P_grid_MW": P_grid,
            "P_cryo_MW": np.full_like(P_dc, self.P_cryo) if np.ndim(P_dc) > 0 else self.P_cryo,
            "P_total_parasitic_MW": self.P_cryo,
            "SOC_new": SOC_new,
            "E_stored_MJ": E_stored_MJ,
            "eta_instantaneous": eta,
            "dE_MJ": dE,
            "mode": mode,
        }
